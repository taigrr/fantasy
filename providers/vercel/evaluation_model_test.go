package vercel

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

func TestEvaluateRoundTrip(t *testing.T) {
	t.Parallel()
	var gotBody map[string]any
	var gotPath, gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		_, _ = w.Write([]byte(`{
			"model": "typesafe-ai/jev",
			"answers": {
				"refund": {"type": "boolean", "probability": 0.98},
				"route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1, "shipping": 0}},
				"quality": {"type": "score", "score": 2.97, "probabilities": {"0": 0, "1": 0, "2": 0.02, "3": 0.98}}
			},
			"usage": {"inputTokens": 275, "outputTokens": 20},
			"providerMetadata": {"gateway": {"cost": "0.00001155", "generationId": "gen_1", "routing": {"finalProvider": "typesafe-ai"}}}
		}`))
	}))
	defer server.Close()

	p, err := New(
		WithBaseURL(server.URL+"/v1"),
		WithAPIKey("gw"),
		WithEvaluationOptions(EvaluationOptions{ZeroDataRetention: true, Only: []string{EvaluationProviderTypeSafe}}),
	)
	require.NoError(t, err)
	ep, ok := p.(fantasy.EvaluationProvider)
	require.True(t, ok, "vercel provider must implement EvaluationProvider")
	model, err := ep.EvaluationModel(t.Context(), "")
	require.NoError(t, err)
	require.Equal(t, Name, model.Provider())
	require.Equal(t, ModelJev, model.Model())

	resp, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State: "I was charged twice for my subscription.",
		Questions: map[string]fantasy.EvaluationQuestion{
			"refund":  fantasy.BoolQuestion("Is the customer asking for money back?"),
			"route":   fantasy.ChoiceQuestion("Route this ticket.", map[string]string{"billing": "payment", "shipping": "delivery"}),
			"quality": fantasy.ScoreQuestion("Rate quality", "poor", "fair", "good", "excellent"),
		},
	})
	require.NoError(t, err)

	require.Equal(t, "/v1"+PathEvaluate, gotPath)
	require.Equal(t, "Bearer gw", gotAuth)
	require.Equal(t, ModelJev, gotBody["model"])
	questions := gotBody["questions"].(map[string]any)
	refund := questions["refund"].(map[string]any)
	require.Equal(t, WireTypeBoolean, refund["type"])
	require.NotContains(t, refund, "criteria")
	gateway := gotBody["providerOptions"].(map[string]any)["gateway"].(map[string]any)
	require.Equal(t, true, gateway["zeroDataRetention"])
	require.Equal(t, []any{EvaluationProviderTypeSafe}, gateway["only"])
	require.NotContains(t, gateway, "type", "wire options must not carry the fantasy type envelope")

	require.Equal(t, ModelJev, resp.Model)
	require.Equal(t, fantasy.EvaluationUsage{InputTokens: 275, OutputTokens: 20, TotalTokens: 295}, resp.Usage)
	require.Equal(t, 0.98, resp.Answers["refund"].Probability)
	require.Equal(t, "billing", resp.Answers["route"].Choice)
	require.Nil(t, resp.Answers["route"].Confidence)
	quality := resp.Answers["quality"]
	require.Equal(t, 2.97, quality.Score)
	require.Equal(t, "excellent", quality.Level())
	require.Equal(t, 0.98, quality.LevelProbabilities[3])
	meta, ok := resp.ProviderMetadata[Name].(*EvaluationMetadata)
	require.True(t, ok)
	require.Equal(t, "0.00001155", meta.Cost)
	require.Equal(t, "gen_1", meta.GenerationID)
	require.Equal(t, EvaluationProviderTypeSafe, meta.Routing.FinalProvider)
}

func TestEvaluationModelWithoutAPIKey(t *testing.T) {
	t.Parallel()
	p, err := New()
	require.NoError(t, err)
	model, err := p.(fantasy.EvaluationProvider).EvaluationModel(t.Context(), ModelJev)
	require.NoError(t, err)
	require.Equal(t, ModelJev, model.Model())
}

func TestEvaluatePerCallOptionsOverrideDefaults(t *testing.T) {
	t.Parallel()
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody = nil
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"model":"m","answers":{},"usage":{}}`))
	}))
	defer server.Close()

	p, _ := New(WithBaseURL(server.URL), WithAPIKey("k"), WithEvaluationOptions(EvaluationOptions{ZeroDataRetention: true}))
	model, err := p.(fantasy.EvaluationProvider).EvaluationModel(t.Context(), ModelJev)
	require.NoError(t, err)

	questions := map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("?")}
	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{State: "x", Questions: questions})
	require.NoError(t, err)
	require.Equal(t, true, gotBody["providerOptions"].(map[string]any)["gateway"].(map[string]any)["zeroDataRetention"])

	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:           "x",
		Questions:       questions,
		ProviderOptions: NewEvaluationOptions(&EvaluationOptions{Order: []string{"a"}}),
	})
	require.NoError(t, err)
	gateway := gotBody["providerOptions"].(map[string]any)["gateway"].(map[string]any)
	require.NotContains(t, gateway, "zeroDataRetention")
	require.Equal(t, []any{"a"}, gateway["order"])
}

func TestChatProviderStillWorks(t *testing.T) {
	t.Parallel()
	p, err := New(WithAPIKey("k"))
	require.NoError(t, err)
	require.Equal(t, Name, p.Name())
	_, err = p.LanguageModel(t.Context(), "anthropic/claude-sonnet-4")
	require.NoError(t, err)
	_, ok := p.(fantasy.EmbeddingProvider)
	require.True(t, ok, "embedding capability must be forwarded")
}
