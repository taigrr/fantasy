// Package vercel provides an implementation of the fantasy AI SDK for Vercel AI Gateway.
package vercel

import (
	"context"
	"fmt"
	"maps"

	"github.com/charmbracelet/openai-go/option"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/openai"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
	sdkOptions           []option.RequestOption
	objectMode           fantasy.ObjectMode
	evaluation           evaluationOptions
}

const (
	// DefaultURL is the default URL for the Vercel AI Gateway API.
	DefaultURL = "https://ai-gateway.vercel.sh/v1"
	// Name is the name of the Vercel provider.
	Name = "vercel"
)

// Option defines a function that configures Vercel provider options.
type Option = func(*options)

// New creates a new Vercel AI Gateway provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		evaluation: newEvaluationOptions(),
		openaiOptions: []openai.Option{
			openai.WithName(Name),
			openai.WithBaseURL(DefaultURL),
		},
		languageModelOptions: []openai.LanguageModelOption{
			openai.WithLanguageModelPrepareCallFunc(languagePrepareModelCall),
			openai.WithLanguageModelUsageFunc(languageModelUsage),
			openai.WithLanguageModelStreamUsageFunc(languageModelStreamUsage),
			openai.WithLanguageModelStreamExtraFunc(languageModelStreamExtra),
			openai.WithLanguageModelExtraContentFunc(languageModelExtraContent),
			openai.WithLanguageModelToPromptFunc(languageModelToPrompt),
		},
		objectMode: fantasy.ObjectModeTool, // Default to tool mode for vercel
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	// Handle object mode: convert unsupported modes to tool
	// Vercel AI Gateway doesn't support native JSON mode, so we use tool or text
	objectMode := providerOptions.objectMode
	if objectMode == fantasy.ObjectModeAuto || objectMode == fantasy.ObjectModeJSON {
		objectMode = fantasy.ObjectModeTool
	}

	providerOptions.openaiOptions = append(
		providerOptions.openaiOptions,
		openai.WithSDKOptions(providerOptions.sdkOptions...),
		openai.WithLanguageModelOptions(providerOptions.languageModelOptions...),
		openai.WithObjectMode(objectMode),
	)
	inner, err := openai.New(providerOptions.openaiOptions...)
	if err != nil {
		return nil, err
	}
	return &provider{Provider: inner, evaluation: providerOptions.evaluation}, nil
}

// provider wraps the OpenAI-compatible chat provider and adds the gateway's
// evaluation modality, which is not exposed on the OpenAI-compatible surface.
type provider struct {
	fantasy.Provider
	evaluation evaluationOptions
}

// EmbeddingModel forwards to the wrapped provider.
func (p *provider) EmbeddingModel(ctx context.Context, modelID string) (fantasy.EmbeddingModel, error) {
	ep, ok := p.Provider.(fantasy.EmbeddingProvider)
	if !ok {
		return nil, fmt.Errorf("%s: embeddings are not supported", p.Name())
	}
	return ep.EmbeddingModel(ctx, modelID)
}

// WithAPIKey sets the API key for the Vercel provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
		o.evaluation.apiKey = apiKey
	}
}

// WithBaseURL sets the base URL for the Vercel provider.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithBaseURL(url))
		o.evaluation.baseURL = url
	}
}

// WithName sets the name for the Vercel provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
		o.evaluation.name = name
	}
}

// WithHeaders sets the headers for the Vercel provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
		maps.Copy(o.evaluation.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the Vercel provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
		o.evaluation.httpClient = client
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithUserAgent(ua))
		o.evaluation.userAgent = ua
	}
}

// WithSDKOptions sets the SDK options for the Vercel provider.
func WithSDKOptions(opts ...option.RequestOption) Option {
	return func(o *options) {
		o.sdkOptions = append(o.sdkOptions, opts...)
	}
}

// WithObjectMode sets the object generation mode for the Vercel provider.
// Supported modes: ObjectModeTool, ObjectModeText.
// ObjectModeAuto and ObjectModeJSON are automatically converted to ObjectModeTool
// since Vercel AI Gateway doesn't support native JSON mode.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}
