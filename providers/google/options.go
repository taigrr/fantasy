// Package google provides an implementation of the fantasy AI SDK for Google's language models.
//
// The Gemini/Vertex client (google.golang.org/genai and its gRPC and
// cloud-auth dependency tree) is only compiled in when built with the
// fantasy_google build tag. Without it, New still succeeds so callers can
// register and configure the provider, but LanguageModel returns
// ErrNotCompiled. Check Enabled to hide the provider up front.
package google

import (
	"cmp"
	"errors"
	"maps"
	"net/http"

	"github.com/google/uuid"
	"github.com/taigrr/fantasy"
)

// ErrNotCompiled is returned by LanguageModel when the binary was built
// without the fantasy_google build tag.
var ErrNotCompiled = errors.New("google: provider not compiled in; rebuild with -tags fantasy_google")

// backend selects between the Gemini API and Vertex AI.
type backend int

const (
	backendUnspecified backend = iota
	backendGeminiAPI
	backendVertexAI
)

// Name is the name of the Google provider.
const Name = "google"

type provider struct {
	options options
}

// ToolCallIDFunc defines a function that generates a tool call ID.
type ToolCallIDFunc = func() string

type options struct {
	apiKey         string
	name           string
	baseURL        string
	headers        map[string]string
	userAgent      string
	client         *http.Client
	backend        backend
	project        string
	location       string
	skipAuth       bool
	toolCallIDFunc ToolCallIDFunc
	objectMode     fantasy.ObjectMode
}

// Option defines a function that configures Google provider options.
type Option = func(*options)

// New creates a new Google provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := options{
		headers: map[string]string{},
		toolCallIDFunc: func() string {
			return uuid.NewString()
		},
	}
	for _, o := range opts {
		o(&options)
	}

	options.name = cmp.Or(options.name, Name)

	return &provider{
		options: options,
	}, nil
}

// WithBaseURL sets the base URL for the Google provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithGeminiAPIKey sets the Gemini API key for the Google provider.
func WithGeminiAPIKey(apiKey string) Option {
	return func(o *options) {
		o.backend = backendGeminiAPI
		o.apiKey = apiKey
		o.project = ""
		o.location = ""
	}
}

// WithVertex configures the Google provider to use Vertex AI.
func WithVertex(project, location string) Option {
	if project == "" || location == "" {
		panic("project and location must be provided")
	}
	return func(o *options) {
		o.backend = backendVertexAI
		o.apiKey = ""
		o.project = project
		o.location = location
	}
}

// WithSkipAuth configures whether to skip authentication for the Google provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithName sets the name for the Google provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithHeaders sets the headers for the Google provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the Google provider.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithToolCallIDFunc sets the function that generates a tool call ID.
func WithToolCallIDFunc(f ToolCallIDFunc) Option {
	return func(o *options) {
		o.toolCallIDFunc = f
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.userAgent = ua
	}
}

// WithObjectMode sets the object generation mode for the Google provider.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}

func (*provider) Name() string {
	return Name
}

// GetReasoningMetadata extracts reasoning metadata from provider options for google models.
func GetReasoningMetadata(providerOptions fantasy.ProviderOptions) *ReasoningMetadata {
	if googleOptions, ok := providerOptions[Name]; ok {
		if reasoning, ok := googleOptions.(*ReasoningMetadata); ok {
			return reasoning
		}
	}
	return nil
}
