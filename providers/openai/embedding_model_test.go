package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

func TestEmbed(t *testing.T) {
	t.Parallel()

	t.Run("returns embeddings in input order", func(t *testing.T) {
		t.Parallel()

		var gotPath string
		var gotBody map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotPath = r.URL.Path
			json.NewDecoder(r.Body).Decode(&gotBody)
			w.Header().Set("Content-Type", "application/json")
			// Return out of order to verify we sort by index.
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"model":  "text-embedding-3-small",
				"data": []map[string]any{
					{"object": "embedding", "index": 1, "embedding": []float64{0.3, 0.4}},
					{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
				},
				"usage": map[string]any{"prompt_tokens": 5, "total_tokens": 5},
			})
		}))
		defer server.Close()

		provider, err := New(WithAPIKey("test"), WithBaseURL(server.URL))
		require.NoError(t, err)

		ep, ok := provider.(fantasy.EmbeddingProvider)
		require.True(t, ok, "openai provider must implement EmbeddingProvider")

		model, err := ep.EmbeddingModel(t.Context(), "text-embedding-3-small")
		require.NoError(t, err)
		require.Equal(t, "openai", model.Provider())
		require.Equal(t, "text-embedding-3-small", model.Model())

		resp, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
			Input:      []string{"first", "second"},
			Dimensions: 2,
		})
		require.NoError(t, err)

		require.Equal(t, "/embeddings", gotPath)
		require.Equal(t, float64(2), gotBody["dimensions"])
		require.Len(t, resp.Embeddings, 2)
		require.Equal(t, fantasy.Embedding{0.1, 0.2}, resp.Embeddings[0])
		require.Equal(t, fantasy.Embedding{0.3, 0.4}, resp.Embeddings[1])
		require.Equal(t, int64(5), resp.Usage.InputTokens)
		require.Equal(t, int64(5), resp.Usage.TotalTokens)
	})

	t.Run("empty input returns empty response without a request", func(t *testing.T) {
		t.Parallel()

		called := false
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			called = true
		}))
		defer server.Close()

		provider, err := New(WithAPIKey("test"), WithBaseURL(server.URL))
		require.NoError(t, err)
		ep := provider.(fantasy.EmbeddingProvider) //nolint:forcetypeassert

		model, err := ep.EmbeddingModel(t.Context(), "text-embedding-3-small")
		require.NoError(t, err)

		resp, err := model.Embed(context.Background(), fantasy.EmbeddingCall{})
		require.NoError(t, err)
		require.Empty(t, resp.Embeddings)
		require.False(t, called)
	})
}
