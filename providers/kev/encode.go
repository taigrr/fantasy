package kev

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/taigrr/fantasy"
)

// Kev reuses five rarely-used Qwen special tokens as structural delimiters
// instead of adding vocabulary. Their roles, in order:
//
//	<|fim_prefix|>  <state>
//	<|fim_middle|>  <q>
//	<|box_start|>   <opt>
//	<|box_end|>     </opt>
//	<|fim_suffix|>  <decide>
var specialTokens = [...]string{"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"}

const (
	// MaxStateTokens caps the state prefix, delimiter included. The state is
	// truncated to fit, as kev.serve does.
	MaxStateTokens = 8192
	// MaxRowTokens caps state plus one question's branch. A branch that does
	// not fit is an error, as in kev.serve; options are never truncated
	// because their end tokens carry the readout.
	MaxRowTokens = 8192

	// NoulOptionNo is the fixed "no" option label for Bool questions.
	NoulOptionNo = "no"
	// NoulOptionYes is the fixed "yes" option label; it is always index 1.
	NoulOptionYes = "yes"
	noulYesIndex  = 1
)

// specialRE matches any Qwen-style <|name|> control token in user text so it
// can be defanged before tokenization.
var specialRE = regexp.MustCompile(`<\|([A-Za-z0-9_]+)\|>`)

// escapeUser rewrites <|name|> to <¦name¦> so callers can never forge a
// delimiter. Mirrors kev/model.py user_tokens.
func escapeUser(text string) string {
	return specialRE.ReplaceAllString(text, "<¦$1¦>")
}

// Tokenizer turns text into token ids without adding BOS/EOS and with
// special-token parsing enabled, and resolves a control token to its id.
type Tokenizer interface {
	Tokenize(text string) []int32
	SpecialToken(text string) (int32, bool)
}

// delimiters holds the resolved ids of the five structural tokens.
type delimiters struct {
	state, question, optStart, optEnd, decide int32
}

func resolveDelimiters(tok Tokenizer) (delimiters, error) {
	var ids [len(specialTokens)]int32
	for i, text := range specialTokens {
		id, ok := tok.SpecialToken(text)
		if !ok {
			return delimiters{}, fmt.Errorf("kev: vocabulary lacks control token %s", text)
		}
		ids[i] = id
	}
	return delimiters{state: ids[0], question: ids[1], optStart: ids[2], optEnd: ids[3], decide: ids[4]}, nil
}

// Row is one causal sequence: the shared state followed by a single
// question's branch. Positions are contiguous from zero. OptEnds and Decide
// index into Tokens.
type Row struct {
	Tokens  []int32
	OptEnds []int
	Decide  int
	// Options are the option labels in wire order, so a probability vector
	// can be mapped back to names.
	Options []string
}

// branch is a rendered question before it is joined with the state.
type branch struct {
	tokens  []int32
	optEnds []int
	options []string
}

// stateTokens renders and truncates the state prefix.
func stateTokens(tok Tokenizer, del delimiters, state string) []int32 {
	body := tok.Tokenize(escapeUser(state))
	if len(body) > MaxStateTokens-1 {
		body = body[:MaxStateTokens-1]
	}
	out := make([]int32, 0, len(body)+1)
	out = append(out, del.state)
	return append(out, body...)
}

// renderBranch builds <q> instr <opt> o0 </opt> ... <decide>.
func renderBranch(tok Tokenizer, del delimiters, instructions string, options []string) branch {
	instr := tok.Tokenize(escapeUser(instructions))
	tokens := make([]int32, 0, 2+len(instr))
	tokens = append(tokens, del.question)
	tokens = append(tokens, instr...)
	optEnds := make([]int, len(options))
	for i, option := range options {
		tokens = append(tokens, del.optStart)
		tokens = append(tokens, tok.Tokenize(escapeUser(option))...)
		tokens = append(tokens, del.optEnd)
		optEnds[i] = len(tokens) - 1
	}
	tokens = append(tokens, del.decide)
	return branch{tokens: tokens, optEnds: optEnds, options: options}
}

// ErrBranchTooLong is returned when state plus a question exceeds MaxRowTokens.
var ErrBranchTooLong = errors.New("kev: question branch too long")

// question is a fantasy question lowered to Kev's option list plus a reader
// that turns the option distribution into an EvaluationAnswer.
type question struct {
	instructions string
	options      []string
	read         func(probs []float64) fantasy.EvaluationAnswer
}

// lowerQuestion maps the three fantasy primitives onto Kev's single
// "pick one option" mechanism, following kev/api.py.
//
// Choice options are emitted in sorted key order. Kev's Python server uses
// the request's JSON insertion order instead; Go maps have none, so sorting
// keeps results deterministic. Option order can shift probabilities
// slightly.
func lowerQuestion(q fantasy.EvaluationQuestion) (question, error) {
	switch q.Type {
	case fantasy.EvaluationQuestionTypeBool:
		options := []string{
			optionText(NoulOptionNo, q.FalseCriteria),
			optionText(NoulOptionYes, q.TrueCriteria),
		}
		return question{
			instructions: q.Instructions,
			options:      options,
			read: func(p []float64) fantasy.EvaluationAnswer {
				return fantasy.EvaluationAnswer{Type: fantasy.EvaluationQuestionTypeBool, Probability: round2(p[noulYesIndex])}
			},
		}, nil
	case fantasy.EvaluationQuestionTypeChoice:
		keys := sortedKeys(q.Options)
		options := make([]string, len(keys))
		for i, key := range keys {
			options[i] = optionText(key, q.Options[key])
		}
		return question{
			instructions: q.Instructions,
			options:      options,
			read: func(p []float64) fantasy.EvaluationAnswer {
				best := argmax(p)
				dist := make(map[string]float64, len(keys))
				for i, key := range keys {
					dist[key] = round2(p[i])
				}
				conf := round2(choiceConfidence(p))
				return fantasy.EvaluationAnswer{
					Type:          fantasy.EvaluationQuestionTypeChoice,
					Choice:        keys[best],
					Probabilities: dist,
					Confidence:    &conf,
				}
			},
		}, nil
	case fantasy.EvaluationQuestionTypeScore:
		levels := append([]string(nil), q.Levels...)
		return question{
			instructions: q.Instructions,
			options:      levels,
			read: func(p []float64) fantasy.EvaluationAnswer {
				score := 0.0
				probs := make([]float64, len(p))
				for i, pi := range p {
					score += float64(i) * pi
					probs[i] = round2(pi)
				}
				conf := round2(scoreConfidence(p))
				return fantasy.EvaluationAnswer{
					Type:               fantasy.EvaluationQuestionTypeScore,
					Score:              round2(score),
					Levels:             levels,
					LevelProbabilities: probs,
					Confidence:         &conf,
				}
			},
		}, nil
	default:
		return question{}, fmt.Errorf("kev: unknown question type %q", q.Type)
	}
}

// optionText renders "name" or "name: description" as kev/api.py does.
func optionText(name, description string) string {
	if description == "" {
		return name
	}
	return name + ": " + description
}

// stateText renders the call state the way kev/api.py render() does:
// strings verbatim, objects as "key: value" lines in field order, arrays as
// "- item" lines.
func stateText(state any) (string, error) {
	if str, ok := state.(string); ok {
		if !utf8.ValidString(str) {
			return "", errors.New("kev: state is not valid UTF-8")
		}
		return str, nil
	}
	ordered, err := toOrdered(state)
	if err != nil {
		return "", err
	}
	return render(ordered, 0), nil
}

// render mirrors kev/api.py render(): field names become labels.
func render(v any, indent int) string {
	pad := strings.Repeat("  ", indent)
	switch val := v.(type) {
	case nil:
		return ""
	case string:
		return val
	case bool:
		if val {
			return "True"
		}
		return "False"
	case json.Number:
		return formatNumber(val)
	case []any:
		lines := make([]string, len(val))
		for i, item := range val {
			lines[i] = pad + "- " + strings.TrimLeft(render(item, indent+1), " ")
		}
		return strings.Join(lines, "\n")
	case *orderedObject:
		lines := make([]string, 0, len(val.keys))
		for _, key := range val.keys {
			child := val.values[key]
			switch child.(type) {
			case *orderedObject, []any:
				lines = append(lines, pad+key+":\n"+render(child, indent+1))
			default:
				lines = append(lines, pad+key+": "+render(child, 0))
			}
		}
		return strings.Join(lines, "\n")
	default:
		return fmt.Sprint(val)
	}
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sortStrings(keys)
	return keys
}

func argmax(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}

// choiceConfidence is (p_max - 1/K) / (1 - 1/K): 0 for uniform, 1 for a
// one-hot distribution.
func choiceConfidence(p []float64) float64 {
	k := float64(len(p))
	if k <= 1 {
		return 1
	}
	uniform := 1 / k
	return (p[argmax(p)] - uniform) / (1 - uniform)
}

// scoreConfidence is 1 - sum(p_i * |i - mode|) / (L - 1): how tightly the
// mass sits around the modal level.
func scoreConfidence(p []float64) float64 {
	levels := len(p)
	if levels <= 1 {
		return 1
	}
	mode := argmax(p)
	spread := 0.0
	for i, pi := range p {
		spread += pi * absInt(i-mode)
	}
	return 1 - spread/float64(levels-1)
}

func absInt(v int) float64 {
	if v < 0 {
		return float64(-v)
	}
	return float64(v)
}
