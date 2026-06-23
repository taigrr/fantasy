package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/taigrr/fantasy"
)

var (
	_ fantasy.EmbeddingProvider = (*provider)(nil)
	_ fantasy.EmbeddingModel    = (*embeddingModel)(nil)
)

type embeddingModel struct {
	provider string
	modelID  string
	client   *bedrockruntime.Client
}

// EmbeddingModel implements fantasy.EmbeddingProvider.
func (p *provider) EmbeddingModel(ctx context.Context, modelID string) (fantasy.EmbeddingModel, error) {
	cfg, err := p.opts.resolveAWSConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to resolve AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		if p.opts.baseURL != "" {
			o.BaseEndpoint = aws.String(p.opts.baseURL)
		}
	})

	return &embeddingModel{
		provider: Name,
		modelID:  modelID,
		client:   client,
	}, nil
}

// Embed implements fantasy.EmbeddingModel.
func (e *embeddingModel) Embed(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	if len(call.Input) == 0 {
		return &fantasy.EmbeddingResponse{}, nil
	}

	switch {
	case strings.HasPrefix(e.modelID, "amazon.titan-embed"):
		return e.embedTitan(ctx, call)
	case strings.HasPrefix(e.modelID, "cohere.embed"):
		return e.embedCohere(ctx, call)
	default:
		return nil, fmt.Errorf("bedrock: embedding model %q is not supported", e.modelID)
	}
}

// invoke performs a single InvokeModel call with a JSON request body
// and decodes the JSON response into out.
func (e *embeddingModel) invoke(ctx context.Context, reqBody any, out any) error {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("bedrock: failed to marshal embedding request: %w", err)
	}
	resp, err := e.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.modelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	})
	if err != nil {
		return fmt.Errorf("bedrock: invoke model %q failed: %w", e.modelID, err)
	}
	if err := json.Unmarshal(resp.Body, out); err != nil {
		return fmt.Errorf("bedrock: failed to decode embedding response: %w", err)
	}
	return nil
}

// titanRequest is the Titan embeddings request body. Titan embeds a
// single text per invocation, so batched inputs are sent sequentially.
type titanRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int64  `json:"dimensions,omitempty"`
	Normalize  bool   `json:"normalize"`
}

type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int64     `json:"inputTextTokenCount"`
}

func (e *embeddingModel) embedTitan(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	embeddings := make([]fantasy.Embedding, 0, len(call.Input))
	var usage fantasy.EmbeddingUsage
	for _, input := range call.Input {
		var out titanResponse
		req := titanRequest{
			InputText:  input,
			Dimensions: call.Dimensions,
			Normalize:  true,
		}
		if err := e.invoke(ctx, req, &out); err != nil {
			return nil, err
		}
		embeddings = append(embeddings, out.Embedding)
		usage.InputTokens += out.InputTextTokenCount
		usage.TotalTokens += out.InputTextTokenCount
	}
	return &fantasy.EmbeddingResponse{Embeddings: embeddings, Usage: usage}, nil
}

// cohereRequest is the Cohere embeddings request body. Cohere supports
// batching, so all inputs are sent in a single invocation.
type cohereRequest struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

type cohereResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (e *embeddingModel) embedCohere(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	var out cohereResponse
	req := cohereRequest{
		Texts:     call.Input,
		InputType: "search_document",
	}
	if err := e.invoke(ctx, req, &out); err != nil {
		return nil, err
	}
	embeddings := make([]fantasy.Embedding, len(out.Embeddings))
	copy(embeddings, out.Embeddings)
	return &fantasy.EmbeddingResponse{Embeddings: embeddings}, nil
}

// Provider implements fantasy.EmbeddingModel.
func (e *embeddingModel) Provider() string { return e.provider }

// Model implements fantasy.EmbeddingModel.
func (e *embeddingModel) Model() string { return e.modelID }
