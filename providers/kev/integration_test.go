package kev

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

// EnvTestBundle points at a local bundle directory (manifest.json, GGUF,
// head.json, reference.json). When unset the integration test is skipped.
const envTestBundle = "KEV_TEST_BUNDLE"

type referenceCase struct {
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
	Answers   map[string]struct {
		Type          string             `json:"type"`
		Noul          float64            `json:"noul"`
		Choice        string             `json:"choice"`
		Score         float64            `json:"score"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"answers"`
	RawProbs [][]float64 `json:"raw_probs"`
}

type referenceQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria"`
}

func toFantasyQuestion(t *testing.T, raw json.RawMessage) fantasy.EvaluationQuestion {
	t.Helper()
	var q referenceQuestion
	require.NoError(t, json.Unmarshal(raw, &q))
	switch q.Type {
	case "noul":
		trueC, falseC := "", ""
		if c, ok := q.Criteria.(map[string]any); ok {
			trueC, _ = c["true"].(string)
			falseC, _ = c["false"].(string)
		}
		return fantasy.BoolQuestionWithCriteria(q.Instructions, trueC, falseC)
	case "choice":
		opts := map[string]string{}
		for k, v := range q.Criteria.(map[string]any) {
			desc, _ := v.(string)
			opts[k] = desc
		}
		return fantasy.ChoiceQuestion(q.Instructions, opts)
	case "score":
		var levels []string
		for _, v := range q.Criteria.([]any) {
			levels = append(levels, v.(string))
		}
		return fantasy.ScoreQuestion(q.Instructions, levels...)
	}
	t.Fatalf("unknown type %q", q.Type)
	return fantasy.EvaluationQuestion{}
}

func TestIntegrationParityWithPython(t *testing.T) {
	bundleDir := os.Getenv(envTestBundle)
	if bundleDir == "" {
		t.Skipf("set %s to a converted bundle directory to run", envTestBundle)
	}
	refData, err := os.ReadFile(filepath.Join(bundleDir, "reference.json"))
	require.NoError(t, err)
	var cases []referenceCase
	require.NoError(t, json.Unmarshal(refData, &cases))
	require.NotEmpty(t, cases)

	p, err := New(WithCheckpoint(Checkpoint(bundleDir)), WithAutoLibraries())
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.(*provider).Close() })
	require.NoError(t, p.(*provider).Load(t.Context()))
	model, err := p.(fantasy.EvaluationProvider).EvaluationModel(t.Context(), "")
	require.NoError(t, err)

	const tolerance = 0.03 // f16 GGUF + llama.cpp kernels vs fp32 torch
	for i, c := range cases {
		questions := map[string]fantasy.EvaluationQuestion{}
		for name, raw := range c.Questions {
			questions[name] = toFantasyQuestion(t, raw)
		}
		var state any = c.State
		var str string
		if json.Unmarshal(c.State, &str) == nil {
			state = str
		}
		resp, err := model.Evaluate(t.Context(), fantasy.EvaluationCall{State: state, Questions: questions})
		require.NoError(t, err, "case %d", i)

		for name, want := range c.Answers {
			got, ok := resp.Answers[name]
			require.True(t, ok, "case %d missing %s", i, name)
			switch want.Type {
			case "noul":
				t.Logf("case %d %s: go=%.3f py=%.3f", i, name, got.Probability, want.Noul)
				require.InDelta(t, want.Noul, got.Probability, tolerance, "case %d %s", i, name)
			case "choice":
				t.Logf("case %d %s: go=%s %v py=%s %v", i, name, got.Choice, got.Probabilities, want.Choice, want.Probabilities)
				require.Equal(t, want.Choice, got.Choice, "case %d %s", i, name)
				for k, p := range want.Probabilities {
					require.InDelta(t, p, got.Probabilities[k], tolerance, "case %d %s[%s]", i, name, k)
				}
			case "score":
				t.Logf("case %d %s: go=%.3f %v py=%.3f %v", i, name, got.Score, got.LevelProbabilities, want.Score, want.Probabilities)
				require.InDelta(t, want.Score, got.Score, tolerance*2, "case %d %s", i, name)
			}
		}
	}
}
