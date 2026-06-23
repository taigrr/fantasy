package fantasy

import "context"

// Embedding is a single embedding vector. Values are float32 because
// that is the de-facto storage format for vector search and keeps
// memory usage half that of float64 with negligible loss for the
// approximate nature of embeddings.
type Embedding = []float32

// EmbeddingCall represents a call to an embedding model.
type EmbeddingCall struct {
	// Input holds one or more texts to embed. Providers that do not
	// support batching natively iterate over the inputs and merge the
	// results so that len(Response.Embeddings) == len(Input) regardless
	// of provider.
	Input []string `json:"input"`

	// Dimensions optionally requests a specific output dimensionality
	// for models that support it (e.g. OpenAI text-embedding-3, Amazon
	// Titan v2). Zero means use the model default.
	Dimensions int64 `json:"dimensions"`

	// UserAgent overrides the provider-level User-Agent header for this
	// call.
	UserAgent string `json:"-"`

	// ProviderOptions holds provider-specific options, keyed by provider
	// id.
	ProviderOptions ProviderOptions `json:"provider_options"`
}

// EmbeddingUsage represents token usage statistics for an embedding call.
type EmbeddingUsage struct {
	InputTokens int64 `json:"input_tokens"`
	TotalTokens int64 `json:"total_tokens"`
}

// EmbeddingResponse represents a response from an embedding model.
type EmbeddingResponse struct {
	// Embeddings holds one vector per input, in the same order as
	// EmbeddingCall.Input.
	Embeddings []Embedding `json:"embeddings"`

	Usage    EmbeddingUsage `json:"usage"`
	Warnings []CallWarning  `json:"warnings"`

	// ProviderMetadata holds provider specific response metadata, keyed
	// by provider id.
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// EmbeddingModel represents a model that converts text into embedding
// vectors.
type EmbeddingModel interface {
	// Embed converts the call's inputs into embedding vectors.
	Embed(context.Context, EmbeddingCall) (*EmbeddingResponse, error)

	// Provider returns the provider id for this model.
	Provider() string
	// Model returns the model id.
	Model() string
}

// EmbeddingProvider is an optional capability interface implemented by
// providers that can construct embedding models. Detect support with a
// type assertion on a [Provider]:
//
//	if ep, ok := p.(fantasy.EmbeddingProvider); ok {
//	    em, err := ep.EmbeddingModel(ctx, "text-embedding-3-small")
//	    // ...
//	}
type EmbeddingProvider interface {
	Provider
	EmbeddingModel(ctx context.Context, modelID string) (EmbeddingModel, error)
}
