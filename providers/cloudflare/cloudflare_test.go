package cloudflare

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/typesafe"
)

const (
	testAccount = "acct123"
	testKey     = "cf-token"
)

const resultBody = `{
	"model": "clef",
	"answers": {
		"urgent": {"type": "noul", "noul": 0.93},
		"team": {"type": "choice", "choice": "technical",
			"probabilities": {"billing": 0.05, "technical": 0.9, "sales": 0.05}, "confidence": 0.85},
		"severity": {"type": "score", "score": 2.4,
			"legend": {"0": "No impact", "1": "Minor", "2": "Major", "3": "Critical"},
			"probabilities": {"0": 0.0, "1": 0.1, "2": 0.4, "3": 0.5}, "confidence": 0.6}
	},
	"usage": {"input_tokens": 120, "output_tokens": 0}
}`

func envelope(result string) string {
	return `{"result":` + result + `,"success":true,"errors":[],"messages":[]}`
}

func newModel(t *testing.T, modelID string, opts ...Option) fantasy.EvaluationModel {
	t.Helper()
	p, err := New(opts...)
	require.NoError(t, err)
	ep, ok := p.(fantasy.EvaluationProvider)
	require.True(t, ok)
	model, err := ep.EvaluationModel(t.Context(), modelID)
	require.NoError(t, err)
	return model
}

var questions = map[string]fantasy.EvaluationQuestion{
	"urgent":   fantasy.BoolQuestion("Is this urgent?"),
	"team":     fantasy.ChoiceQuestion("Which team?", map[string]string{"billing": "Payments", "technical": "Outages", "sales": ""}),
	"severity": fantasy.ScoreQuestion("How severe?", "No impact", "Minor", "Major", "Critical"),
}

func TestNewRequiresAccountOrBaseURL(t *testing.T) {
	t.Parallel()
	_, err := New(WithAPIKey("k"))
	require.ErrorIs(t, err, ErrAccountRequired)
	_, err = New(WithAccountID(testAccount))
	require.NoError(t, err)
	_, err = New(WithBaseURL("http://localhost/run"))
	require.NoError(t, err)
}

func TestEvaluateRoundTripViaAccount(t *testing.T) {
	t.Parallel()
	var gotPath, gotAuth, gotGateway string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotGateway = r.Header.Get(HeaderGatewayID)
		body, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(body, &gotBody))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(envelope(resultBody)))
	}))
	defer server.Close()

	model := newModel(t, "", WithAPIURL(server.URL), WithAccountID(testAccount), WithAPIKey(testKey), WithGatewayID("default"))
	require.Equal(t, Name, model.Provider())
	require.Equal(t, ModelClef, model.Model())

	resp, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     "Checkout has been failing for every customer for the last hour.",
		Questions: questions,
	})
	require.NoError(t, err)

	require.Equal(t, "/accounts/"+testAccount+"/ai/run/@cf/cloudflare/clef", gotPath)
	require.Equal(t, "Bearer "+testKey, gotAuth)
	require.Equal(t, "default", gotGateway)
	require.Equal(t, ModelClef, gotBody["model"])
	_, hasImages := gotBody["images"]
	require.False(t, hasImages)
	team := gotBody["questions"].(map[string]any)["team"].(map[string]any)
	require.Equal(t, typesafe.WireTypeChoice, team["type"])
	require.Nil(t, team["criteria"].(map[string]any)["sales"])

	require.Equal(t, ModelClef, resp.Model)
	require.Equal(t, fantasy.EvaluationUsage{InputTokens: 120, TotalTokens: 120}, resp.Usage)
	require.Equal(t, 0.93, resp.Answers["urgent"].Probability)
	require.Equal(t, "technical", resp.Answers["team"].Choice)
	require.Equal(t, 0.85, *resp.Answers["team"].Confidence)
	severity := resp.Answers["severity"]
	require.Equal(t, 2.4, severity.Score)
	require.Equal(t, "Major", severity.Level())
	require.Equal(t, []float64{0, 0.1, 0.4, 0.5}, severity.LevelProbabilities)
}

func TestEvaluateBaseURLOverrideAndPrefixedModel(t *testing.T) {
	t.Parallel()
	var gotPath string
	var gotBody WireRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(envelope(`{"model":"clef-flash","answers":{"urgent":{"type":"noul","noul":0.2}},"usage":{"input_tokens":1,"output_tokens":0}}`)))
	}))
	defer server.Close()

	model := newModel(t, ModelPrefix+ModelClefFlash, WithBaseURL(server.URL+"/v1/acct/gw/workers-ai/"), WithAPIKey(testKey))
	require.Equal(t, ModelClefFlash, model.Model())
	_, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     map[string]any{"invoice": map[string]any{"total": 1250}},
		Questions: map[string]fantasy.EvaluationQuestion{"urgent": fantasy.BoolQuestion("?")},
	})
	require.NoError(t, err)
	require.Equal(t, "/v1/acct/gw/workers-ai/@cf/cloudflare/clef-flash", gotPath)
	require.Equal(t, ModelClefFlash, gotBody.Model)
	require.JSONEq(t, `{"invoice":{"total":1250}}`, string(gotBody.State))
}

func TestEvaluateImages(t *testing.T) {
	t.Parallel()
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(envelope(`{"model":"clef","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}`)))
	}))
	defer server.Close()

	image := Image{ContentType: ImageContentTypePNG, Base64: "AAAA"}
	model := newModel(t, ModelClef, WithBaseURL(server.URL), WithEvaluationOptions(EvaluationOptions{Images: []Image{image}}))
	_, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     "receipt",
		Questions: map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("Is it legible?")},
	})
	require.NoError(t, err)
	images := gotBody["images"].([]any)
	require.Len(t, images, 1)
	require.Equal(t, ImageContentTypePNG, images[0].(map[string]any)["content_type"])

	// Per-call options replace provider defaults.
	perCall := NewEvaluationOptions(&EvaluationOptions{Images: []Image{image, image}})
	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:           "receipt",
		Questions:       map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("Is it legible?")},
		ProviderOptions: perCall,
	})
	require.NoError(t, err)
	require.Len(t, gotBody["images"].([]any), 2)

	tooMany := make([]Image, MaxImages+1)
	_, err = EncodeRequest(fantasy.EvaluationCall{State: "s"}, ModelClef, &EvaluationOptions{Images: tooMany})
	require.ErrorContains(t, err, "exceeds the limit")
}

func TestEvaluateEnvelopeFailure(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"result":null,"success":false,"errors":[{"code":7000,"message":"No route for that URI"}],"messages":[]}`))
	}))
	defer server.Close()

	model := newModel(t, ModelClef, WithBaseURL(server.URL))
	_, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     "x",
		Questions: map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("?")},
	})
	require.ErrorIs(t, err, ErrRequestFailed)
	require.ErrorContains(t, err, "7000: No route for that URI")
}

func TestEvaluateHTTPError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"messages":[]}`))
	}))
	defer server.Close()

	model := newModel(t, ModelClef, WithBaseURL(server.URL), WithAPIKey("bad"))
	_, err := model.Evaluate(context.Background(), fantasy.EvaluationCall{})
	require.ErrorContains(t, err, "invalid call")

	_, err = model.Evaluate(context.Background(), fantasy.EvaluationCall{
		State:     "x",
		Questions: map[string]fantasy.EvaluationQuestion{"q": fantasy.BoolQuestion("?")},
	})
	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr)
	require.Equal(t, http.StatusUnauthorized, providerErr.StatusCode)
	require.Equal(t, "10000: Authentication error", providerErr.Message)
}

func TestLanguageModelUnsupported(t *testing.T) {
	t.Parallel()
	p, err := New(WithAccountID(testAccount))
	require.NoError(t, err)
	_, err = p.LanguageModel(t.Context(), ModelClef)
	require.ErrorIs(t, err, ErrLanguageModelUnsupported)
}

func TestNormalizeModelID(t *testing.T) {
	t.Parallel()
	require.Equal(t, ModelClef, NormalizeModelID(""))
	require.Equal(t, ModelClef, NormalizeModelID(" clef "))
	require.Equal(t, ModelClefFlash, NormalizeModelID(ModelPrefix+ModelClefFlash))
	require.Equal(t, "/@cf/cloudflare/clef-flash", ModelPath(ModelClefFlash))
	require.Equal(t, DefaultAPIURL+"/accounts/a/ai/run", RunURL(DefaultAPIURL+"/", "a"))
}

func TestEvaluationOptionsRoundTrip(t *testing.T) {
	t.Parallel()
	opts := fantasy.ProviderOptions{Name: &EvaluationOptions{Images: []Image{{ContentType: ImageContentTypeWebP, Base64: "QQ=="}}}}
	data, err := json.Marshal(opts)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	decoded, err := fantasy.UnmarshalProviderOptions(raw)
	require.NoError(t, err)
	require.Equal(t, "QQ==", decoded[Name].(*EvaluationOptions).Images[0].Base64)
}
