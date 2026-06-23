package openai

import (
	"context"
	"fmt"
	"sort"

	"github.com/charmbracelet/openai-go"
	"github.com/charmbracelet/openai-go/option"
	"github.com/charmbracelet/openai-go/packages/param"
	"github.com/taigrr/fantasy"
)

var (
	_ fantasy.EmbeddingProvider = (*provider)(nil)
	_ fantasy.EmbeddingModel    = (*embeddingModel)(nil)
)

type embeddingModel struct {
	provider string
	modelID  string
	client   openai.Client
}

func newEmbeddingModel(modelID, provider string, client openai.Client) *embeddingModel {
	return &embeddingModel{
		provider: provider,
		modelID:  modelID,
		client:   client,
	}
}

// EmbeddingModel implements fantasy.EmbeddingProvider.
func (o *provider) EmbeddingModel(_ context.Context, modelID string) (fantasy.EmbeddingModel, error) {
	return newEmbeddingModel(modelID, o.options.name, o.newClient()), nil
}

// Embed implements fantasy.EmbeddingModel.
func (e *embeddingModel) Embed(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	if len(call.Input) == 0 {
		return &fantasy.EmbeddingResponse{}, nil
	}

	params := openai.EmbeddingNewParams{
		Model:          e.modelID,
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: call.Input},
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if call.Dimensions > 0 {
		params.Dimensions = param.NewOpt(call.Dimensions)
	}

	requestOpts := []option.RequestOption{}
	if call.UserAgent != "" {
		requestOpts = append(requestOpts, option.WithHeader("User-Agent", call.UserAgent))
	}

	resp, err := e.client.Embeddings.New(ctx, params, requestOpts...)
	if err != nil {
		return nil, fmt.Errorf("openai: embedding request failed: %w", err)
	}

	// The API returns embeddings tagged with their input index; sort by
	// index so the output order matches call.Input regardless of how the
	// API ordered the response.
	data := append([]openai.Embedding(nil), resp.Data...)
	sort.Slice(data, func(i, j int) bool { return data[i].Index < data[j].Index })

	embeddings := make([]fantasy.Embedding, len(data))
	for i, d := range data {
		vec := make(fantasy.Embedding, len(d.Embedding))
		for j, v := range d.Embedding {
			vec[j] = float32(v)
		}
		embeddings[i] = vec
	}

	return &fantasy.EmbeddingResponse{
		Embeddings: embeddings,
		Usage: fantasy.EmbeddingUsage{
			InputTokens: resp.Usage.PromptTokens,
			TotalTokens: resp.Usage.TotalTokens,
		},
	}, nil
}

// Provider implements fantasy.EmbeddingModel.
func (e *embeddingModel) Provider() string { return e.provider }

// Model implements fantasy.EmbeddingModel.
func (e *embeddingModel) Model() string { return e.modelID }
