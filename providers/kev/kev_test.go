package kev

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

// fakeTokenizer maps each whitespace-separated word to a stable id and
// recognises the five delimiters.
type fakeTokenizer struct {
	ids map[string]int32
}

func newFakeTokenizer() *fakeTokenizer {
	tok := &fakeTokenizer{ids: map[string]int32{}}
	for i, s := range specialTokens {
		tok.ids[s] = int32(1000 + i)
	}
	return tok
}

func (f *fakeTokenizer) Tokenize(text string) []int32 {
	var out []int32
	for word := range strings.FieldsSeq(text) {
		id, ok := f.ids[word]
		if !ok {
			id = int32(len(f.ids) + 1)
			f.ids[word] = id
		}
		out = append(out, id)
	}
	return out
}

func (f *fakeTokenizer) SpecialToken(text string) (int32, bool) {
	id, ok := f.ids[text]
	return id, ok
}

func TestEscapeUser(t *testing.T) {
	t.Parallel()
	require.Equal(t, "a <¦fim_prefix¦> b <¦box_end¦>", escapeUser("a <|fim_prefix|> b <|box_end|>"))
	require.Equal(t, "plain <| not a token |>", escapeUser("plain <| not a token |>"))
}

func TestRowLayout(t *testing.T) {
	t.Parallel()
	tok := newFakeTokenizer()
	del, err := resolveDelimiters(tok)
	require.NoError(t, err)

	state := stateTokens(tok, del, "hello world")
	require.Equal(t, del.state, state[0])
	require.Len(t, state, 3)

	br := renderBranch(tok, del, "pick one", []string{"a b", "c"})
	row := joinRow(state, br)

	// <state> hello world <q> pick one <opt> a b </opt> <opt> c </opt> <decide>
	require.Equal(t, del.question, row.Tokens[3])
	require.Equal(t, del.decide, row.Tokens[len(row.Tokens)-1])
	require.Equal(t, len(row.Tokens)-1, row.Decide)
	require.Len(t, row.OptEnds, 2)
	for _, end := range row.OptEnds {
		require.Equal(t, del.optEnd, row.Tokens[end])
	}
	require.Equal(t, del.optStart, row.Tokens[row.OptEnds[0]-3])
	require.Equal(t, del.optStart, row.Tokens[row.OptEnds[1]-2])
	require.Equal(t, []string{"a b", "c"}, row.Options)
}

func TestStateTruncation(t *testing.T) {
	t.Parallel()
	tok := newFakeTokenizer()
	del, _ := resolveDelimiters(tok)
	words := make([]string, MaxStateTokens+50)
	for i := range words {
		words[i] = "w" + string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('a'+(i/676)%26))
	}
	state := stateTokens(tok, del, strings.Join(words, " "))
	require.Len(t, state, MaxStateTokens)
}

func TestLowerQuestion(t *testing.T) {
	t.Parallel()

	boolQ, err := lowerQuestion(fantasy.BoolQuestionWithCriteria("urgent?", "time-sensitive", ""))
	require.NoError(t, err)
	require.Equal(t, []string{"no", "yes: time-sensitive"}, boolQ.options)
	ans := boolQ.read([]float64{0.12, 0.88})
	require.Equal(t, fantasy.EvaluationQuestionTypeBool, ans.Type)
	require.Equal(t, 0.88, ans.Probability)

	choiceQ, err := lowerQuestion(fantasy.ChoiceQuestion("team?", map[string]string{"shipping": "delays", "billing": "", "returns": "refunds"}))
	require.NoError(t, err)
	require.Equal(t, []string{"billing", "returns: refunds", "shipping: delays"}, choiceQ.options)
	ans = choiceQ.read([]float64{0.25, 0.47, 0.28})
	require.Equal(t, "returns", ans.Choice)
	require.Equal(t, map[string]float64{"billing": 0.25, "returns": 0.47, "shipping": 0.28}, ans.Probabilities)
	require.InDelta(t, 0.205, *ans.Confidence, 0.006)

	scoreQ, err := lowerQuestion(fantasy.ScoreQuestion("how mad?", "Calm", "Frustrated", "Very angry"))
	require.NoError(t, err)
	require.Equal(t, []string{"Calm", "Frustrated", "Very angry"}, scoreQ.options)
	ans = scoreQ.read([]float64{0.0, 0.56, 0.44})
	require.Equal(t, 1.44, ans.Score)
	require.Equal(t, "Frustrated", ans.Level())
	require.Equal(t, []float64{0, 0.56, 0.44}, ans.LevelProbabilities)
	require.InDelta(t, 0.78, *ans.Confidence, 0.005)

	_, err = lowerQuestion(fantasy.EvaluationQuestion{Type: "weird"})
	require.Error(t, err)
}

func TestConfidenceFormulas(t *testing.T) {
	t.Parallel()
	require.InDelta(t, 0.0, choiceConfidence([]float64{1.0 / 3, 1.0 / 3, 1.0 / 3}), 1e-9)
	require.InDelta(t, 1.0, choiceConfidence([]float64{0, 1, 0}), 1e-9)
	require.Equal(t, 1.0, choiceConfidence([]float64{1}))
	require.InDelta(t, 1.0, scoreConfidence([]float64{0, 1, 0}), 1e-9)
	require.InDelta(t, 0.5, scoreConfidence([]float64{0.5, 0, 0.5}), 1e-9)
}

func TestStateText(t *testing.T) {
	t.Parallel()
	text, err := stateText("plain")
	require.NoError(t, err)
	require.Equal(t, "plain", text)

	// Raw JSON keeps key order, matching Python's dict rendering exactly.
	text, err = stateText(json.RawMessage(`{"messages":[{"role":"user","content":"hi, can you change my email address?"}]}`))
	require.NoError(t, err)
	require.Equal(t, "messages:\n  - role: user\n    content: hi, can you change my email address?", text)

	text, err = stateText(json.RawMessage(`{"a":[1,{"b":2}],"c":true,"d":1.5,"e":null}`))
	require.NoError(t, err)
	require.Equal(t, "a:\n  - 1\n  - b: 2\nc: True\nd: 1.5\ne: ", text)

	// Structs keep declaration order.
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	text, err = stateText(struct {
		Messages []msg `json:"messages"`
	}{Messages: []msg{{Role: "user", Content: "hi"}}})
	require.NoError(t, err)
	require.Equal(t, "messages:\n  - role: user\n    content: hi", text)

	// Go maps are sorted by encoding/json.
	text, err = stateText(map[string]any{"count": 2, "ok": true})
	require.NoError(t, err)
	require.Equal(t, "count: 2\nok: True", text)

	text, err = stateText([]string{"a", "b"})
	require.NoError(t, err)
	require.Equal(t, "- a\n- b", text)
}

func TestHead(t *testing.T) {
	t.Parallel()
	head, err := ParseHead([]byte(`{
		"format": 1, "hidden_size": 2, "head_dim": 2, "temperature": 1,
		"q_weight": [[1,0],[0,1]], "q_bias": [0,0],
		"k_weight": [[1,0],[0,1]], "k_bias": [0,0]
	}`))
	require.NoError(t, err)

	decide := []float32{1, 0}
	options := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	logits, err := head.Logits(decide, options)
	require.NoError(t, err)
	scale := 1 / math.Sqrt(2)
	require.InDelta(t, scale, logits[0], 1e-9)
	require.InDelta(t, 0, logits[1], 1e-9)
	require.InDelta(t, -scale, logits[2], 1e-9)

	probs, err := head.Probabilities(decide, options)
	require.NoError(t, err)
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	require.InDelta(t, 1, sum, 1e-9)
	require.Greater(t, probs[0], probs[1])
	require.Greater(t, probs[1], probs[2])

	head.Temperature = 1000
	flat, _ := head.Probabilities(decide, options)
	require.InDelta(t, flat[0], flat[2], 1e-3)

	_, err = head.Logits([]float32{1}, options)
	require.Error(t, err)
	_, err = ParseHead([]byte(`{"format": 2}`))
	require.Error(t, err)
	_, err = ParseHead([]byte(`{"format": 1, "hidden_size": 2, "head_dim": 1, "q_weight": [[1]], "q_bias": [0], "k_weight": [[1]], "k_bias": [0]}`))
	require.Error(t, err, "rows shorter than hidden_size must be rejected")
}
