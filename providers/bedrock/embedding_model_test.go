package bedrock

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

// newEmbeddingProvider builds a Bedrock provider that targets the given
// TLS test server. A TLS server is required because the Bedrock runtime
// signs with a bearer token when AWS_BEARER_TOKEN_BEDROCK is present,
// and bearer auth refuses plain HTTP. The server's client trusts its
// self-signed cert.
func newEmbeddingProvider(t *testing.T, srv *httptest.Server) fantasy.EmbeddingProvider {
	t.Helper()
	p, err := New(
		WithRegion("us-east-1"),
		WithAPIKey("test-bearer-token"),
		WithBaseURL(srv.URL),
		WithAWSConfig(aws.Config{
			Region:      "us-east-1",
			Credentials: credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
			HTTPClient:  srv.Client(),
		}),
	)
	require.NoError(t, err)
	ep, ok := p.(fantasy.EmbeddingProvider)
	require.True(t, ok, "bedrock provider must implement EmbeddingProvider")
	return ep
}

func TestEmbedTitan(t *testing.T) {
	t.Parallel()

	var paths []string
	var bodies []map[string]any
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embedding":           []float32{0.1, 0.2, 0.3},
			"inputTextTokenCount": 2,
		})
	}))
	defer server.Close()

	ep := newEmbeddingProvider(t, server)
	model, err := ep.EmbeddingModel(t.Context(), "amazon.titan-embed-text-v2:0")
	require.NoError(t, err)
	require.Equal(t, "bedrock", model.Provider())
	require.Equal(t, "amazon.titan-embed-text-v2:0", model.Model())

	resp, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
		Input:      []string{"alpha", "beta"},
		Dimensions: 3,
	})
	require.NoError(t, err)

	// Titan embeds one input per call, so two inputs => two invocations.
	require.Len(t, paths, 2)
	require.Contains(t, paths[0], "amazon.titan-embed-text-v2:0")
	require.Equal(t, float64(3), bodies[0]["dimensions"])
	require.Equal(t, true, bodies[0]["normalize"])
	require.Equal(t, "alpha", bodies[0]["inputText"])

	require.Len(t, resp.Embeddings, 2)
	require.Equal(t, fantasy.Embedding{0.1, 0.2, 0.3}, resp.Embeddings[0])
	require.Equal(t, int64(4), resp.Usage.InputTokens)
}

func TestEmbedCohere(t *testing.T) {
	t.Parallel()

	var body map[string]any
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float32{{0.1, 0.2}, {0.3, 0.4}},
		})
	}))
	defer server.Close()

	ep := newEmbeddingProvider(t, server)
	model, err := ep.EmbeddingModel(t.Context(), "cohere.embed-english-v3")
	require.NoError(t, err)

	resp, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
		Input: []string{"alpha", "beta"},
	})
	require.NoError(t, err)

	// Cohere batches: one invocation for all inputs.
	require.Equal(t, []any{"alpha", "beta"}, body["texts"])
	require.Equal(t, "search_document", body["input_type"])
	require.Len(t, resp.Embeddings, 2)
	require.Equal(t, fantasy.Embedding{0.3, 0.4}, resp.Embeddings[1])
}

func TestEmbedUnsupportedModel(t *testing.T) {
	t.Parallel()

	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	ep := newEmbeddingProvider(t, server)
	model, err := ep.EmbeddingModel(t.Context(), "anthropic.claude-3")
	require.NoError(t, err)

	_, err = model.Embed(context.Background(), fantasy.EmbeddingCall{Input: []string{"x"}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "not supported")
}
