// Package cloudflare provides a fantasy evaluation provider for Cloudflare's
// Clef decision models served by Workers AI.
//
// Clef speaks the System One wire protocol (the same request and answer
// shapes as TypeSafe's Jev), but Workers AI addresses the model in the URL
// and wraps every response in the standard Cloudflare API envelope
// ({"result": ..., "success": true, "errors": [...]}). This package handles
// both and otherwise reuses the typesafe encoder and decoder.
//
// Requests are posted to <run root>/@cf/cloudflare/<model>. The run root is
// derived from the account id by default
// (https://api.cloudflare.com/client/v4/accounts/<id>/ai/run) and can be
// replaced wholesale with WithBaseURL, for example to route through an AI
// Gateway.
//
// Clef models are evaluation-only: the provider implements
// fantasy.EvaluationProvider and LanguageModel returns an error.
package cloudflare

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/evalhttp"
	"github.com/taigrr/fantasy/providers/typesafe"
)

const (
	// Name is the provider name.
	Name = "cloudflare"
	// DefaultAPIURL is the Cloudflare REST API base URL.
	DefaultAPIURL = "https://api.cloudflare.com/client/v4"

	// ModelClef is the 27B Clef model.
	ModelClef = "clef"
	// ModelClefFlash is the 9B latency-optimised Clef model.
	ModelClefFlash = "clef-flash"
	// ModelPrefix is the Workers AI namespace Clef models live under; model
	// ids with or without it are accepted.
	ModelPrefix = "@cf/cloudflare/"

	// HeaderGatewayID routes a request through a named AI Gateway.
	HeaderGatewayID = "cf-aig-gateway-id"

	// TypeEvaluationOptions is the global type identifier for per-call
	// evaluation options.
	TypeEvaluationOptions = Name + ".evaluation.options"

	// ImageContentTypePNG is an accepted embedded image content type.
	ImageContentTypePNG = "image/png"
	// ImageContentTypeJPEG is an accepted embedded image content type.
	ImageContentTypeJPEG = "image/jpeg"
	// ImageContentTypeWebP is an accepted embedded image content type.
	ImageContentTypeWebP = "image/webp"

	// MaxImages is the most images Clef accepts in one request.
	MaxImages = 4
)

// ErrLanguageModelUnsupported is returned by LanguageModel: Clef answers
// typed questions and does not generate text.
var ErrLanguageModelUnsupported = errors.New("cloudflare: language models are not supported; use EvaluationModel")

// ErrAccountRequired is returned by New when neither an account id nor a
// base URL is configured.
var ErrAccountRequired = errors.New("cloudflare: account id (or base URL) is required")

// ErrRequestFailed is returned when Workers AI answers 2xx with
// success=false.
var ErrRequestFailed = errors.New("cloudflare: request was not successful")

var (
	_ fantasy.EvaluationProvider = (*provider)(nil)
	_ fantasy.EvaluationModel    = (*evaluationModel)(nil)
)

func init() {
	fantasy.RegisterProviderType(TypeEvaluationOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v EvaluationOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

type options struct {
	baseURL    string
	apiURL     string
	accountID  string
	gatewayID  string
	apiKey     string
	name       string
	headers    map[string]string
	userAgent  string
	httpClient evalhttp.HTTPDoer
	retry      fantasy.RetryOptions
	defaults   *EvaluationOptions
}

// Option configures the Cloudflare provider.
type Option = func(*options)

type provider struct {
	options options
}

// New creates a Cloudflare Workers AI provider. Either WithAccountID or
// WithBaseURL must be given.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		apiURL:  DefaultAPIURL,
		headers: map[string]string{},
		retry:   fantasy.DefaultRetryOptions(),
	}
	for _, opt := range opts {
		opt(&providerOptions)
	}
	providerOptions.name = cmp.Or(providerOptions.name, Name)
	if providerOptions.baseURL == "" {
		if providerOptions.accountID == "" {
			return nil, ErrAccountRequired
		}
		providerOptions.baseURL = RunURL(providerOptions.apiURL, providerOptions.accountID)
	}
	if providerOptions.gatewayID != "" {
		providerOptions.headers[HeaderGatewayID] = providerOptions.gatewayID
	}
	return &provider{options: providerOptions}, nil
}

// RunURL returns the Workers AI run root for an account under the given API
// base URL.
func RunURL(apiURL, accountID string) string {
	return strings.TrimRight(apiURL, "/") + "/accounts/" + accountID + "/ai/run"
}

// WithAPIKey sets the bearer token (a Cloudflare API token with Workers AI
// permission).
func WithAPIKey(apiKey string) Option {
	return func(o *options) { o.apiKey = apiKey }
}

// WithAccountID sets the Cloudflare account whose Workers AI is used.
func WithAccountID(accountID string) Option {
	return func(o *options) { o.accountID = accountID }
}

// WithAPIURL replaces the Cloudflare REST API origin used to derive the run
// root from the account id. It is ignored when WithBaseURL is set.
func WithAPIURL(apiURL string) Option {
	return func(o *options) { o.apiURL = apiURL }
}

// WithBaseURL sets the Workers AI run root directly; "/@cf/cloudflare/<model>"
// is appended to it. Use this to point at an AI Gateway or any other server
// exposing the Workers AI contract. When set, the account id is not needed.
func WithBaseURL(baseURL string) Option {
	return func(o *options) { o.baseURL = baseURL }
}

// WithGatewayID routes requests through the named AI Gateway on the account.
func WithGatewayID(gatewayID string) Option {
	return func(o *options) { o.gatewayID = gatewayID }
}

// WithName overrides the provider name reported on models and metadata.
func WithName(name string) Option {
	return func(o *options) { o.name = name }
}

// WithHeaders adds extra headers to every request.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) { maps.Copy(o.headers, headers) }
}

// WithUserAgent sets the User-Agent header.
func WithUserAgent(userAgent string) Option {
	return func(o *options) { o.userAgent = userAgent }
}

// WithHTTPClient sets the underlying HTTP client.
func WithHTTPClient(client evalhttp.HTTPDoer) Option {
	return func(o *options) { o.httpClient = client }
}

// WithRetryOptions configures retry behaviour.
func WithRetryOptions(retry fantasy.RetryOptions) Option {
	return func(o *options) { o.retry = retry }
}

// WithEvaluationOptions sets default per-call options applied to every
// evaluation unless the call carries its own.
func WithEvaluationOptions(opts EvaluationOptions) Option {
	return func(o *options) { o.defaults = &opts }
}

// Image is an embedded image sent alongside the state. Clef accepts PNG,
// JPEG and WebP, up to MaxImages per request; remote URLs are not accepted.
type Image struct {
	// ContentType is one of the ImageContentType constants.
	ContentType string `json:"content_type"`
	// Base64 is the standard base64 encoding of the image bytes.
	Base64 string `json:"base64"`
}

// EvaluationOptions are Clef-specific per-call options.
type EvaluationOptions struct {
	// Images are placed before the state. This is a Clef extension to the
	// System One API.
	Images []Image `json:"images,omitempty"`
}

// Options implements fantasy.ProviderOptionsData.
func (*EvaluationOptions) Options() {}

// MarshalJSON implements json.Marshaler.
func (o EvaluationOptions) MarshalJSON() ([]byte, error) {
	type plain EvaluationOptions
	return fantasy.MarshalProviderType(TypeEvaluationOptions, plain(o))
}

// UnmarshalJSON implements json.Unmarshaler.
func (o *EvaluationOptions) UnmarshalJSON(data []byte) error {
	type plain EvaluationOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = EvaluationOptions(p)
	return nil
}

// NewEvaluationOptions wraps options for EvaluationCall.ProviderOptions.
func NewEvaluationOptions(opts *EvaluationOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{Name: opts}
}

// Name implements fantasy.Provider.
func (p *provider) Name() string { return p.options.name }

// LanguageModel implements fantasy.Provider and always fails.
func (p *provider) LanguageModel(context.Context, string) (fantasy.LanguageModel, error) {
	return nil, ErrLanguageModelUnsupported
}

// EvaluationModel implements fantasy.EvaluationProvider. The model id may be
// given with or without the ModelPrefix; an empty id selects ModelClef.
func (p *provider) EvaluationModel(_ context.Context, modelID string) (fantasy.EvaluationModel, error) {
	return &evaluationModel{
		provider: p.options.name,
		modelID:  NormalizeModelID(modelID),
		defaults: p.options.defaults,
		client: &evalhttp.Client{
			BaseURL:    p.options.baseURL,
			APIKey:     p.options.apiKey,
			Headers:    p.options.headers,
			UserAgent:  p.options.userAgent,
			HTTPClient: p.options.httpClient,
			Retry:      p.options.retry,
		},
	}, nil
}

// NormalizeModelID strips the Workers AI namespace and surrounding
// whitespace, defaulting to ModelClef.
func NormalizeModelID(modelID string) string {
	modelID = strings.TrimSpace(modelID)
	modelID = strings.TrimPrefix(modelID, ModelPrefix)
	return cmp.Or(modelID, ModelClef)
}

// ModelPath returns the request path for a (normalised) model id, relative
// to the run root.
func ModelPath(modelID string) string {
	return "/" + ModelPrefix + NormalizeModelID(modelID)
}

type evaluationModel struct {
	provider string
	modelID  string
	defaults *EvaluationOptions
	client   *evalhttp.Client
}

// Provider implements fantasy.EvaluationModel.
func (m *evaluationModel) Provider() string { return m.provider }

// Model implements fantasy.EvaluationModel.
func (m *evaluationModel) Model() string { return m.modelID }

// Evaluate implements fantasy.EvaluationModel.
func (m *evaluationModel) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, fmt.Errorf("%s: invalid call: %w", m.provider, err)
	}
	extra := m.defaults
	if raw, ok := call.ProviderOptions[m.provider]; ok {
		if opts, ok := raw.(*EvaluationOptions); ok {
			extra = opts
		}
	}
	wireReq, err := EncodeRequest(call, m.modelID, extra)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.provider, err)
	}
	var envelope Envelope
	if err := m.client.PostJSON(ctx, evalhttp.Request{Path: ModelPath(m.modelID), Body: wireReq, UserAgent: call.UserAgent}, &envelope); err != nil {
		return nil, err
	}
	if !envelope.Success {
		return nil, fmt.Errorf("%s: %w: %s", m.provider, ErrRequestFailed, envelope.Errors.String())
	}
	var wireResp typesafe.WireResponse
	if err := json.Unmarshal(envelope.Result, &wireResp); err != nil {
		return nil, fmt.Errorf("%s: decode result: %w", m.provider, err)
	}
	resp, err := typesafe.DecodeResponse(wireResp, call.Questions, m.provider)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.provider, err)
	}
	return resp, nil
}

// WireRequest is the on-the-wire request shape: the System One request plus
// Clef's optional images.
type WireRequest struct {
	typesafe.WireRequest
	Images []Image `json:"images,omitempty"`
}

// EncodeRequest converts an EvaluationCall into the Clef wire format. The
// model id is normalised to the short form Workers AI expects in the body.
func EncodeRequest(call fantasy.EvaluationCall, modelID string, opts *EvaluationOptions) (WireRequest, error) {
	base, err := typesafe.EncodeRequest(call, NormalizeModelID(modelID))
	if err != nil {
		return WireRequest{}, err
	}
	wire := WireRequest{WireRequest: base}
	if opts != nil {
		if len(opts.Images) > MaxImages {
			return WireRequest{}, fmt.Errorf("%d images exceeds the limit of %d", len(opts.Images), MaxImages)
		}
		wire.Images = opts.Images
	}
	return wire, nil
}

// Message is an entry in the Cloudflare API envelope's errors or messages.
type Message struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message"`
}

// Messages is a list of envelope messages.
type Messages []Message

// String joins the messages for display.
func (m Messages) String() string {
	if len(m) == 0 {
		return "no error details"
	}
	parts := make([]string, 0, len(m))
	for _, msg := range m {
		if msg.Code != 0 {
			parts = append(parts, fmt.Sprintf("%d: %s", msg.Code, msg.Message))
			continue
		}
		parts = append(parts, msg.Message)
	}
	return strings.Join(parts, "; ")
}

// Envelope is the standard Cloudflare API response wrapper.
type Envelope struct {
	Result   json.RawMessage `json:"result"`
	Success  bool            `json:"success"`
	Errors   Messages        `json:"errors"`
	Messages Messages        `json:"messages"`
}
