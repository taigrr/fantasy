package typesafe

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

func newModel(t *testing.T, url string, opts ...Option) fantasy.EvaluationModel {
	t.Helper()
	p, err := New(append([]Option{WithBaseURL(url)}, opts...)...)
	require.NoError(t, err)
	ep, ok := p.(fantasy.EvaluationProvider)
	require.True(t, ok)
	model, err := ep.EvaluationModel(t.Context(), "")
	require.NoError(t, err)
	return model
}

func TestEvaluateRoundTrip(t *testing.T) {
	t.Parallel()
	var gotBody map[string]any
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, PathSystemOne, r.URL.Path)
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(body, &gotBody))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "jev-1.13.0",
			"answers": {
				"is_urgent": {"type": "noul", "noul": 0.95},
				"department": {"type": "choice", "choice": "billing",
					"probabilities": {"billing": 0.88, "technical": 0.12}, "confidence": 0.81},
				"frustration": {"type": "score", "score": 1.05,
					"legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
					"probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92}
			},
			"usage": {"input_tokens": 296, "output_tokens": 20},
			"latency_ms": 123
		}`))
	}))
	defer server.Close()

	model := newModel(t, server.URL, WithAPIKey("k"))
	require.Equal(t, Name, model.Provider())
	require.Equal(t, ModelJevLatest, model.Model())

	resp, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State: "Help! My payouts have been failing for 3 days.",
		Questions: map[string]fantasy.EvaluationQuestion{
			"is_urgent":   fantasy.BoolQuestionWithCriteria("Does this convey urgency?", "Explicitly time-sensitive", "No urgency"),
			"department":  fantasy.ChoiceQuestion("Which team?", map[string]string{"billing": "Payments", "technical": ""}),
			"frustration": fantasy.ScoreQuestion("How frustrated?", "Calm", "Frustrated", "Very angry"),
		},
	})
	require.NoError(t, err)

	require.Equal(t, "Bearer k", gotAuth)
	require.Equal(t, ModelJevLatest, gotBody["model"])
	require.Equal(t, "Help! My payouts have been failing for 3 days.", gotBody["state"])
	questions := gotBody["questions"].(map[string]any)
	urgent := questions["is_urgent"].(map[string]any)
	require.Equal(t, WireTypeNoul, urgent["type"])
	require.Equal(t, "Explicitly time-sensitive", urgent["criteria"].(map[string]any)["true"])
	dept := questions["department"].(map[string]any)["criteria"].(map[string]any)
	require.Nil(t, dept["technical"])
	require.Equal(t, "Payments", dept["billing"])
	require.Equal(t, []any{"Calm", "Frustrated", "Very angry"}, questions["frustration"].(map[string]any)["criteria"])

	require.Equal(t, ModelJev1_13_0, resp.Model)
	require.Equal(t, fantasy.EvaluationUsage{InputTokens: 296, OutputTokens: 20, TotalTokens: 316}, resp.Usage)
	require.Equal(t, 0.95, resp.Answers["is_urgent"].Probability)
	require.True(t, resp.Answers["is_urgent"].Yes())
	require.Equal(t, "billing", resp.Answers["department"].Choice)
	require.Equal(t, 0.81, *resp.Answers["department"].Confidence)
	frustration := resp.Answers["frustration"]
	require.Equal(t, 1.05, frustration.Score)
	require.Equal(t, "Frustrated", frustration.Level())
	require.Equal(t, []float64{0, 0.95, 0.05}, frustration.LevelProbabilities)
	meta, ok := resp.ProviderMetadata[Name].(*ProviderMetadata)
	require.True(t, ok)
	require.Equal(t, 123.0, meta.LatencyMS)
}

func TestEvaluateStructuredState(t *testing.T) {
	t.Parallel()
	var gotBody WireRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"model":"x","answers":{"q":{"type":"noul","noul":0.1}},"usage":{}}`))
	}))
	defer server.Close()

	p, _ := New(WithBaseURL(server.URL))
	model, err := p.(fantasy.EvaluationProvider).EvaluationModel(t.Context(), ModelJevPreview)
	require.NoError(t, err)
	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     []map[string]string{{"role": "user", "content": "hi"}},
		Questions: map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("?")},
	})
	require.NoError(t, err)
	require.Equal(t, ModelJevPreview, gotBody.Model)
	require.JSONEq(t, `[{"content":"hi","role":"user"}]`, string(gotBody.State))
}

func TestEvaluateErrors(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer server.Close()

	model := newModel(t, server.URL, WithAPIKey("bad"))
	_, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{})
	require.ErrorContains(t, err, "invalid call")

	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     "x",
		Questions: map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("?")},
	})
	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr)
	require.Equal(t, http.StatusUnauthorized, providerErr.StatusCode)
	require.Equal(t, "bad key", providerErr.Message)
}

func TestLanguageModelUnsupported(t *testing.T) {
	t.Parallel()
	p, err := New()
	require.NoError(t, err)
	_, err = p.LanguageModel(t.Context(), "jev-latest")
	require.ErrorIs(t, err, ErrLanguageModelUnsupported)
}

func TestModelsKevShape(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, PathModels, r.URL.Path)
		_, _ = w.Write([]byte(`{"models":[{"id":"kev-latest","aliases":["jev-latest"],"run":"jaredpalmer/kev-4b"}]}`))
	}))
	defer server.Close()

	p, _ := New(WithBaseURL(server.URL))
	models, raw, err := Models(context.Background(), p)
	require.NoError(t, err)
	require.Len(t, models, 1)
	require.Equal(t, ModelKevLatest, models[0].ID)
	require.Equal(t, []string{"jev-latest"}, models[0].Aliases)
	require.NotEmpty(t, raw)
}

func TestDecodeAnswerErrors(t *testing.T) {
	t.Parallel()
	_, err := DecodeAnswer(WireAnswer{Type: WireTypeNoul}, fantasy.EvaluationQuestion{})
	require.Error(t, err)
	_, err = DecodeAnswer(WireAnswer{Type: WireTypeScore}, fantasy.EvaluationQuestion{})
	require.Error(t, err)
	_, err = DecodeAnswer(WireAnswer{Type: "mystery"}, fantasy.EvaluationQuestion{})
	require.Error(t, err)

	score := 0.5
	ans, err := DecodeAnswer(WireAnswer{Type: WireTypeScore, Score: &score}, fantasy.ScoreQuestion("r", "a", "b"))
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, ans.Levels)
}

func TestProviderMetadataRoundTrip(t *testing.T) {
	t.Parallel()
	meta := fantasy.ProviderMetadata{Name: &ProviderMetadata{LatencyMS: 42}}
	data, err := json.Marshal(meta)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	decoded, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	require.Equal(t, 42.0, decoded[Name].(*ProviderMetadata).LatencyMS)
}
