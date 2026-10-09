package clef

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/taigrr/fantasy"
)

// The prompt layout follows joint_schema_model.py of the Clef release. Each
// piece is tokenized on its own, then concatenated, exactly as the model was
// trained; the head reads the mean hidden state over the instructions span
// of each question and over the semantics span of each option.
const (
	systemPrompt = "Read the complete state and schema. Decide every field jointly. Each answer " +
		"must be exactly one of that field's allowed options."
	prefixText       = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\nSTATE:\n"
	schemaHeaderText = "\n\nSCHEMA FIELDS:\n"
	allowedText      = "\nALLOWED OPTIONS:\n"
	endFieldText     = "END FIELD\n"
	suffixText       = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"

	// WireTypeNoul is Clef's name for Bool questions.
	WireTypeNoul = "noul"
	// WireTypeChoice is the name for Choice questions.
	WireTypeChoice = "choice"
	// WireTypeScore is the name for Score questions.
	WireTypeScore = "score"

	// OptionTrue and OptionFalse are the fixed option ids of a Bool question.
	OptionTrue  = "true"
	OptionFalse = "false"

	defaultTrueDescription  = "The proposition is true or the answer is yes."
	defaultFalseDescription = "The proposition is false or the answer is no."

	// MaxQuestions is the most questions one call may carry, as on Workers AI.
	MaxQuestions = 64
)

// ErrSchemaTooLong is returned when the prompt without the state already
// exceeds the context: the state is truncated to fit, questions never are.
var ErrSchemaTooLong = errors.New("clef: questions do not fit in the context")

// Tokenizer turns text into token ids without adding BOS/EOS and with
// special-token parsing enabled.
type Tokenizer interface {
	Tokenize(text string) []int32
}

// option is one allowed answer of a lowered question.
type option struct {
	id          string
	description *string
}

// question is a fantasy question lowered to Clef's option list.
type question struct {
	name         string
	wireType     string
	order        llama.DecisionOrder
	instructions string
	options      []option
	// read turns this question's option probabilities into an answer.
	read func(probs []float64) fantasy.EvaluationAnswer
}

// prompt is a rendered request ready for the engine.
type prompt struct {
	tokens []int32
	orders []llama.DecisionOrder
	// questions are in prompt order; option scores come back in the same
	// order, question by question.
	questions []question
}

// lowerQuestion maps the three fantasy primitives onto Clef's option list,
// following question_options() and systemone_answer() in
// joint_schema_model.py. Choice options are sorted by id, as Clef does.
func lowerQuestion(name string, q fantasy.EvaluationQuestion) (question, error) {
	instructions := q.Instructions
	if instructions == "" {
		instructions = name
	}
	switch q.Type {
	case fantasy.EvaluationQuestionTypeBool:
		trueDesc, falseDesc := defaultTrueDescription, defaultFalseDescription
		if q.TrueCriteria != "" {
			trueDesc = q.TrueCriteria
		}
		if q.FalseCriteria != "" {
			falseDesc = q.FalseCriteria
		}
		return question{
			name: name, wireType: WireTypeNoul, order: llama.DecisionOrderQuestionNoul, instructions: instructions,
			options: []option{{id: OptionTrue, description: &trueDesc}, {id: OptionFalse, description: &falseDesc}},
			read: func(p []float64) fantasy.EvaluationAnswer {
				return fantasy.EvaluationAnswer{Type: fantasy.EvaluationQuestionTypeBool, Probability: round4(p[0])}
			},
		}, nil
	case fantasy.EvaluationQuestionTypeChoice:
		keys := make([]string, 0, len(q.Options))
		for key := range q.Options {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		options := make([]option, len(keys))
		for i, key := range keys {
			options[i] = option{id: key}
			if desc := q.Options[key]; desc != "" {
				options[i].description = &desc
			}
		}
		return question{
			name: name, wireType: WireTypeChoice, order: llama.DecisionOrderQuestionChoice, instructions: instructions,
			options: options,
			read: func(p []float64) fantasy.EvaluationAnswer {
				best := argmax(p)
				dist := make(map[string]float64, len(keys))
				for i, key := range keys {
					dist[key] = round4(p[i])
				}
				conf := round4(p[best])
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
		options := make([]option, len(levels))
		for i := range levels {
			desc := levels[i]
			options[i] = option{id: strconv.Itoa(i), description: &desc}
		}
		return question{
			name: name, wireType: WireTypeScore, order: llama.DecisionOrderQuestionScore, instructions: instructions,
			options: options,
			read: func(p []float64) fantasy.EvaluationAnswer {
				score := 0.0
				probs := make([]float64, len(p))
				for i, pi := range p {
					score += float64(i) * pi
					probs[i] = round4(pi)
				}
				conf := round4(p[argmax(p)])
				return fantasy.EvaluationAnswer{
					Type:               fantasy.EvaluationQuestionTypeScore,
					Score:              round4(score),
					Levels:             levels,
					LevelProbabilities: probs,
					Confidence:         &conf,
				}
			},
		}, nil
	default:
		return question{}, fmt.Errorf("clef: unknown question type %q", q.Type)
	}
}

// semantics renders an option as Clef does: compact JSON with sorted keys,
// the description omitted when absent.
func (o option) semantics() string {
	var buf bytes.Buffer
	buf.WriteByte('{')
	if o.description != nil {
		buf.WriteString(`"description":`)
		writePythonJSONString(&buf, *o.description)
		buf.WriteByte(',')
	}
	buf.WriteString(`"option_id":`)
	writePythonJSONString(&buf, o.id)
	buf.WriteByte('}')
	return buf.String()
}

// renderPrompt tokenizes the request piece by piece. Questions are laid out
// in the given order; maxTokens bounds the whole prompt and only the state
// is truncated to fit.
func renderPrompt(tok Tokenizer, stateText string, questions []question, maxTokens int) (prompt, error) {
	var schema []int32
	var orders []llama.DecisionOrder
	add := func(text string, order llama.DecisionOrder) int {
		ids := tok.Tokenize(text)
		schema = append(schema, ids...)
		for range ids {
			orders = append(orders, order)
		}
		return len(ids)
	}
	add(schemaHeaderText, llama.DecisionOrderNone)
	for i, q := range questions {
		add(fmt.Sprintf("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", i+1, q.name, q.wireType), llama.DecisionOrderNone)
		if add(q.instructions, q.order) == 0 {
			return prompt{}, fmt.Errorf("clef: question %q: instructions produce no tokens", q.name)
		}
		add(allowedText, llama.DecisionOrderNone)
		for j, opt := range q.options {
			add(fmt.Sprintf("OPTION %d: ", j+1), llama.DecisionOrderNone)
			if add(opt.semantics(), llama.DecisionOrderOption) == 0 {
				return prompt{}, fmt.Errorf("clef: question %q: option %q produces no tokens", q.name, opt.id)
			}
			add("\n", llama.DecisionOrderNone)
		}
		add(endFieldText, llama.DecisionOrderNone)
	}

	prefix := tok.Tokenize(prefixText)
	suffix := tok.Tokenize(suffixText)
	fixed := len(prefix) + len(schema) + len(suffix)
	if fixed > maxTokens {
		return prompt{}, fmt.Errorf("%w: %d tokens before the state, maximum is %d", ErrSchemaTooLong, fixed, maxTokens)
	}
	state := tok.Tokenize(stateText)
	if len(state) > maxTokens-fixed {
		state = state[:maxTokens-fixed]
	}

	tokens := make([]int32, 0, fixed+len(state))
	tokens = append(tokens, prefix...)
	tokens = append(tokens, state...)
	tokens = append(tokens, schema...)
	tokens = append(tokens, suffix...)
	all := make([]llama.DecisionOrder, 0, len(tokens))
	for range prefix {
		all = append(all, llama.DecisionOrderNone)
	}
	for range state {
		all = append(all, llama.DecisionOrderNone)
	}
	all = append(all, orders...)
	for range suffix {
		all = append(all, llama.DecisionOrderNone)
	}
	return prompt{tokens: tokens, orders: all, questions: questions}, nil
}

// stateText renders the call state as Clef's render() does: strings
// verbatim, anything else as compact JSON with sorted keys.
func stateText(state any) (string, error) {
	if str, ok := state.(string); ok {
		if !utf8.ValidString(str) {
			return "", errors.New("clef: state is not valid UTF-8")
		}
		return str, nil
	}
	var buf bytes.Buffer
	if err := writePythonJSON(&buf, state); err != nil {
		return "", fmt.Errorf("clef: render state: %w", err)
	}
	return buf.String(), nil
}

// writePythonJSON writes v the way Python's
// json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
// does. Go floats that are whole numbers keep a ".0" like Python floats;
// json.Number and json.RawMessage keep their literal text.
func writePythonJSON(buf *bytes.Buffer, v any) error {
	switch val := v.(type) {
	case nil:
		buf.WriteString("null")
	case bool:
		if val {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case string:
		writePythonJSONString(buf, val)
	case json.Number:
		buf.WriteString(val.String())
	case json.RawMessage:
		var decoded any
		dec := json.NewDecoder(bytes.NewReader(val))
		dec.UseNumber()
		if err := dec.Decode(&decoded); err != nil {
			return err
		}
		return writePythonJSON(buf, decoded)
	case float32:
		writePythonFloat(buf, float64(val))
	case float64:
		writePythonFloat(buf, val)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		fmt.Fprint(buf, val)
	case []any:
		buf.WriteByte('[')
		for i, item := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writePythonJSON(buf, item); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(val))
		for key := range val {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		buf.WriteByte('{')
		for i, key := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			writePythonJSONString(buf, key)
			buf.WriteByte(':')
			if err := writePythonJSON(buf, val[key]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		rv := reflect.ValueOf(v)
		switch rv.Kind() { //nolint:exhaustive // other kinds fall through to encoding/json
		case reflect.Slice, reflect.Array:
			items := make([]any, rv.Len())
			for i := range items {
				items[i] = rv.Index(i).Interface()
			}
			return writePythonJSON(buf, items)
		case reflect.Map:
			if rv.Type().Key().Kind() == reflect.String {
				m := make(map[string]any, rv.Len())
				iter := rv.MapRange()
				for iter.Next() {
					m[iter.Key().String()] = iter.Value().Interface()
				}
				return writePythonJSON(buf, m)
			}
		case reflect.Pointer, reflect.Interface:
			if rv.IsNil() {
				buf.WriteString("null")
				return nil
			}
			return writePythonJSON(buf, rv.Elem().Interface())
		}
		// Structs and anything else go through encoding/json, then back
		// through this writer so keys are sorted and numbers kept literal.
		data, err := json.Marshal(v)
		if err != nil {
			return err
		}
		return writePythonJSON(buf, json.RawMessage(data))
	}
	return nil
}

func writePythonFloat(buf *bytes.Buffer, f float64) {
	switch {
	case math.IsNaN(f):
		buf.WriteString("NaN")
	case math.IsInf(f, 1):
		buf.WriteString("Infinity")
	case math.IsInf(f, -1):
		buf.WriteString("-Infinity")
	default:
		s := strconv.FormatFloat(f, 'g', -1, 64)
		if !strings.ContainsAny(s, ".e") {
			s += ".0"
		}
		buf.WriteString(s)
	}
}

// writePythonJSONString escapes as json.dumps(ensure_ascii=False) does:
// only quotes, backslashes and control characters.
func writePythonJSONString(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			buf.WriteString(`\"`)
		case '\\':
			buf.WriteString(`\\`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		case '\b':
			buf.WriteString(`\b`)
		case '\f':
			buf.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(buf, `\u%04x`, r)
				continue
			}
			buf.WriteRune(r)
		}
	}
	buf.WriteByte('"')
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

// round4 rounds to four decimals, as systemone_answer() does.
func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

// softmax converts option scores to probabilities.
func softmax(scores []float32) []float64 {
	out := make([]float64, len(scores))
	maxScore := math.Inf(-1)
	for _, s := range scores {
		maxScore = math.Max(maxScore, float64(s))
	}
	sum := 0.0
	for i, s := range scores {
		out[i] = math.Exp(float64(s) - maxScore)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
