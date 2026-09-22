package fantasy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEvaluationQuestionValidate(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		q       EvaluationQuestion
		wantErr bool
	}{
		{"bool ok", BoolQuestion("Is it urgent?"), false},
		{"bool no instructions", BoolQuestion(""), true},
		{"choice ok", ChoiceQuestion("Route?", map[string]string{"a": "", "b": "desc"}), false},
		{"choice empty", ChoiceQuestion("Route?", nil), true},
		{"score ok", ScoreQuestion("Rate", "low", "high"), false},
		{"score too few", ScoreQuestion("Rate", "only"), true},
		{"unknown type", EvaluationQuestion{Type: "weird", Instructions: "x"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := tt.q.Validate()
			require.Equal(t, tt.wantErr, err != nil, "err=%v", err)
		})
	}
	tooMany := make(map[string]string, EvaluationMaxChoiceOptions+1)
	for i := 0; i <= EvaluationMaxChoiceOptions; i++ {
		tooMany[string(rune('a'+i%26))+string(rune('a'+i/26))] = ""
	}
	require.Error(t, ChoiceQuestion("x", tooMany).Validate())
}

func TestEvaluationCallValidate(t *testing.T) {
	t.Parallel()
	require.Error(t, (EvaluationCall{}).Validate())
	require.Error(t, (EvaluationCall{State: "x"}).Validate())
	require.Error(t, (EvaluationCall{State: "x", Questions: map[string]EvaluationQuestion{"bad": BoolQuestion("")}}).Validate())
	require.NoError(t, (EvaluationCall{State: map[string]any{"a": 1}, Questions: map[string]EvaluationQuestion{"ok": BoolQuestion("q")}}).Validate())
}

func TestEvaluationAnswerHelpers(t *testing.T) {
	t.Parallel()
	require.True(t, (EvaluationAnswer{Probability: 0.7}).Yes())
	require.False(t, (EvaluationAnswer{Probability: 0.2}).Yes())

	ans := EvaluationAnswer{Score: 1.44, Levels: []string{"Calm", "Frustrated", "Very angry"}}
	require.Equal(t, "Frustrated", ans.Level())
	ans.Score = 2.9
	require.Equal(t, "Very angry", ans.Level())
	ans.Score = -1
	require.Equal(t, "Calm", ans.Level())
	require.Equal(t, "", (EvaluationAnswer{}).Level())

	require.InDelta(t, 0.6, (EvaluationAnswer{Probabilities: map[string]float64{"a": 0.8, "b": 0.2}}).Margin(), 1e-9)
	require.InDelta(t, 0.1, (EvaluationAnswer{Probabilities: map[string]float64{"a": 0.5, "b": 0.4, "c": 0.1}}).Margin(), 1e-9)
	require.InDelta(t, 1.0, (EvaluationAnswer{Probabilities: map[string]float64{"a": 1}}).Margin(), 1e-9)
	require.Equal(t, 0.0, (EvaluationAnswer{}).Margin())
}
