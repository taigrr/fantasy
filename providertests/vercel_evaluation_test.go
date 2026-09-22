package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/vercel"
)

func TestVercelEvaluation(t *testing.T) {
	r := vcr.NewRecorder(t)
	provider, err := vercel.New(
		vercel.WithAPIKey(os.Getenv("FANTASY_VERCEL_API_KEY")),
		vercel.WithHTTPClient(&http.Client{Transport: r}),
	)
	require.NoError(t, err)

	ep, ok := provider.(fantasy.EvaluationProvider)
	require.True(t, ok)
	model, err := ep.EvaluationModel(t.Context(), vercel.ModelJev)
	require.NoError(t, err)

	resp, err := model.Evaluate(t.Context(), fantasy.EvaluationCall{
		State: "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. Fix this today.",
		Questions: map[string]fantasy.EvaluationQuestion{
			"department": fantasy.ChoiceQuestion("Which team should handle this?", map[string]string{
				"returns":  "Exchanges, refunds, wrong or damaged items",
				"shipping": "Delivery status, delays, lost packages",
				"billing":  "Charges, invoices, payment problems",
			}),
			"escalate":    fantasy.BoolQuestion("Does this need urgent human attention?"),
			"frustration": fantasy.ScoreQuestion("How frustrated is the customer?", "Calm", "Frustrated", "Very angry"),
		},
	})
	require.NoError(t, err)

	require.NotEmpty(t, resp.Model)
	require.Positive(t, resp.Usage.InputTokens)

	department := resp.Answers["department"]
	require.Equal(t, fantasy.EvaluationQuestionTypeChoice, department.Type)
	require.Contains(t, []string{"returns", "shipping", "billing"}, department.Choice)
	require.Len(t, department.Probabilities, 3)
	sum := 0.0
	for _, p := range department.Probabilities {
		sum += p
	}
	require.InDelta(t, 1.0, sum, 0.02)

	escalate := resp.Answers["escalate"]
	require.Equal(t, fantasy.EvaluationQuestionTypeBool, escalate.Type)
	require.True(t, escalate.Yes(), "explicit 'fix this today' should read as urgent, got p=%.3f", escalate.Probability)

	frustration := resp.Answers["frustration"]
	require.Equal(t, fantasy.EvaluationQuestionTypeScore, frustration.Type)
	require.Len(t, frustration.LevelProbabilities, 3)
	require.Equal(t, []string{"Calm", "Frustrated", "Very angry"}, frustration.Levels)
	require.GreaterOrEqual(t, frustration.Score, 1.0, "customer is clearly not calm")

	meta, ok := resp.ProviderMetadata[vercel.Name].(*vercel.EvaluationMetadata)
	require.True(t, ok)
	require.NotEmpty(t, meta.GenerationID)
}
