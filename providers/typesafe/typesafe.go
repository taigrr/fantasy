// Package typesafe provides a fantasy provider for TypeSafe AI's System One
// decision models (Jev) via the native HTTP API at api.typesafe.ai.
//
// Any server that speaks the same contract can be targeted with
// WithBaseURL; the open-weight Kev server (github.com/jaredpalmer/kev) is
// wire-compatible and needs no API key.
//
// TypeSafe models are evaluation-only: the provider implements
// fantasy.EvaluationProvider and LanguageModel returns an error.
package typesafe

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/evalhttp"
)

const (
	// Name is the provider name.
	Name = "typesafe"
	// DefaultURL is the TypeSafe API base URL.
	DefaultURL = "https://api.typesafe.ai"

	// PathSystemOne is the evaluation endpoint.
	PathSystemOne = "/v1/systemone"
	// PathModels lists available models.
	PathModels = "/v1/models"

	// ModelJevLatest is the alias for the current flagship model.
	ModelJevLatest = "jev-latest"
	// ModelJevPreview is the alias for the preview model.
	ModelJevPreview = "jev-preview"
	// ModelJev1_13_0 is the pinned current Jev release.
	ModelJev1_13_0 = "jev-1.13.0"
	// ModelKevLatest is the alias a local Kev server advertises.
	ModelKevLatest = "kev-latest"

	// WireTypeNoul is TypeSafe's wire name for Bool questions.
	WireTypeNoul = "noul"
	// WireTypeChoice is the wire name for Choice questions.
	WireTypeChoice = "choice"
	// WireTypeScore is the wire name for Score questions.
	WireTypeScore = "score"
)

// ErrLanguageModelUnsupported is returned by LanguageModel: TypeSafe models
// answer typed questions and do not generate text.
var ErrLanguageModelUnsupported = errors.New("typesafe: language models are not supported; use EvaluationModel")

var (
	_ fantasy.EvaluationProvider = (*provider)(nil)
	_ fantasy.EvaluationModel    = (*evaluationModel)(nil)
)

type options struct {
	baseURL    string
	apiKey     string
	name       string
	headers    map[string]string
	userAgent  string
	httpClient evalhttp.HTTPDoer
	retry      fantasy.RetryOptions
}

// Option configures the TypeSafe provider.
type Option = func(*options)

type provider struct {
	options options
}

// New creates a TypeSafe provider.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		headers: map[string]string{},
		retry:   fantasy.DefaultRetryOptions(),
	}
	for _, opt := range opts {
		opt(&providerOptions)
	}
	providerOptions.baseURL = cmp.Or(providerOptions.baseURL, DefaultURL)
	providerOptions.name = cmp.Or(providerOptions.name, Name)
	return &provider{options: providerOptions}, nil
}

// WithAPIKey sets the bearer token. Local Kev servers need none.
func WithAPIKey(apiKey string) Option {
	return func(o *options) { o.apiKey = apiKey }
}

// WithBaseURL points the provider at a different server, e.g. a local Kev.
func WithBaseURL(baseURL string) Option {
	return func(o *options) { o.baseURL = baseURL }
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

// WithHTTPClient sets the underlying HTTP client. Any type with a Do method
// matching *http.Client works, including the OpenAI SDK's option.HTTPClient.
func WithHTTPClient(client evalhttp.HTTPDoer) Option {
	return func(o *options) { o.httpClient = client }
}

// WithRetryOptions configures retry behaviour.
func WithRetryOptions(retry fantasy.RetryOptions) Option {
	return func(o *options) { o.retry = retry }
}

// Name implements fantasy.Provider.
func (p *provider) Name() string { return p.options.name }

// LanguageModel implements fantasy.Provider and always fails.
func (p *provider) LanguageModel(context.Context, string) (fantasy.LanguageModel, error) {
	return nil, ErrLanguageModelUnsupported
}

// EvaluationModel implements fantasy.EvaluationProvider.
func (p *provider) EvaluationModel(_ context.Context, modelID string) (fantasy.EvaluationModel, error) {
	return &evaluationModel{
		provider: p.options.name,
		modelID:  cmp.Or(modelID, ModelJevLatest),
		client:   p.newClient(),
	}, nil
}

func (p *provider) newClient() *evalhttp.Client {
	return &evalhttp.Client{
		BaseURL:    p.options.baseURL,
		APIKey:     p.options.apiKey,
		Headers:    p.options.headers,
		UserAgent:  p.options.userAgent,
		HTTPClient: p.options.httpClient,
		Retry:      p.options.retry,
	}
}

// ModelInfo describes a model returned by GET /v1/models. TypeSafe reports
// Name; Kev reports ID and Aliases.
type ModelInfo struct {
	ID          string   `json:"id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Aliases     []string `json:"aliases,omitempty"`
	Alias       string   `json:"alias,omitempty"`
	AliasFor    string   `json:"alias_for,omitempty"`
	Description string   `json:"description,omitempty"`
	ReleaseDate string   `json:"release_date,omitempty"`
}

// Models lists the models the server exposes. The raw payload is returned
// alongside the parsed list because implementations differ in shape.
func Models(ctx context.Context, p fantasy.Provider) ([]ModelInfo, json.RawMessage, error) {
	tp, ok := p.(*provider)
	if !ok {
		return nil, nil, fmt.Errorf("typesafe: provider %q is not a typesafe provider", p.Name())
	}
	var raw json.RawMessage
	if err := tp.newClient().GetJSON(ctx, PathModels, &raw); err != nil {
		return nil, nil, err
	}
	var list []ModelInfo
	if err := json.Unmarshal(raw, &list); err == nil {
		return list, raw, nil
	}
	var envelope struct {
		Models []ModelInfo `json:"models"`
		Data   []ModelInfo `json:"data"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, raw, fmt.Errorf("typesafe: unexpected models payload: %w", err)
	}
	if len(envelope.Models) > 0 {
		return envelope.Models, raw, nil
	}
	return envelope.Data, raw, nil
}

type evaluationModel struct {
	provider string
	modelID  string
	client   *evalhttp.Client
}

// Provider implements fantasy.EvaluationModel.
func (m *evaluationModel) Provider() string { return m.provider }

// Model implements fantasy.EvaluationModel.
func (m *evaluationModel) Model() string { return m.modelID }

// Evaluate implements fantasy.EvaluationModel.
func (m *evaluationModel) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, fmt.Errorf("typesafe: invalid call: %w", err)
	}
	wireReq, err := EncodeRequest(call, m.modelID)
	if err != nil {
		return nil, fmt.Errorf("typesafe: %w", err)
	}
	var wireResp WireResponse
	if err := m.client.PostJSON(ctx, evalhttp.Request{Path: PathSystemOne, Body: wireReq, UserAgent: call.UserAgent}, &wireResp); err != nil {
		return nil, err
	}
	resp, err := DecodeResponse(wireResp, call.Questions, m.provider)
	if err != nil {
		return nil, fmt.Errorf("typesafe: %w", err)
	}
	return resp, nil
}

// WireQuestion is the on-the-wire question shape.
type WireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// WireRequest is the on-the-wire request shape.
type WireRequest struct {
	State     json.RawMessage         `json:"state"`
	Model     string                  `json:"model"`
	Questions map[string]WireQuestion `json:"questions"`
}

// WireAnswer is the on-the-wire answer shape.
type WireAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// WireUsage is the on-the-wire usage shape.
type WireUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// WireResponse is the on-the-wire response shape.
type WireResponse struct {
	Model     string                `json:"model"`
	Answers   map[string]WireAnswer `json:"answers"`
	Usage     WireUsage             `json:"usage"`
	LatencyMS *float64              `json:"latency_ms,omitempty"`
}

// EncodeRequest converts an EvaluationCall into the TypeSafe wire format.
func EncodeRequest(call fantasy.EvaluationCall, model string) (WireRequest, error) {
	state, err := json.Marshal(call.State)
	if err != nil {
		return WireRequest{}, fmt.Errorf("marshal state: %w", err)
	}
	questions := make(map[string]WireQuestion, len(call.Questions))
	for name, question := range call.Questions {
		wireQuestion, err := EncodeQuestion(question)
		if err != nil {
			return WireRequest{}, fmt.Errorf("question %q: %w", name, err)
		}
		questions[name] = wireQuestion
	}
	return WireRequest{State: state, Model: model, Questions: questions}, nil
}

// EncodeQuestion converts a single question into the TypeSafe wire format.
func EncodeQuestion(question fantasy.EvaluationQuestion) (WireQuestion, error) {
	switch question.Type {
	case fantasy.EvaluationQuestionTypeBool:
		wire := WireQuestion{Type: WireTypeNoul, Instructions: question.Instructions}
		if criteria := BoolCriteria(question); len(criteria) > 0 {
			wire.Criteria = criteria
		}
		return wire, nil
	case fantasy.EvaluationQuestionTypeChoice:
		return WireQuestion{Type: WireTypeChoice, Instructions: question.Instructions, Criteria: ChoiceCriteria(question)}, nil
	case fantasy.EvaluationQuestionTypeScore:
		return WireQuestion{Type: WireTypeScore, Instructions: question.Instructions, Criteria: question.Levels}, nil
	default:
		return WireQuestion{}, fmt.Errorf("unknown question type %q", question.Type)
	}
}

// BoolCriteria builds the optional {true, false} criteria map for a Bool
// question. It returns nil when neither description is set.
func BoolCriteria(question fantasy.EvaluationQuestion) map[string]string {
	if question.TrueCriteria == "" && question.FalseCriteria == "" {
		return nil
	}
	criteria := map[string]string{}
	if question.TrueCriteria != "" {
		criteria["true"] = question.TrueCriteria
	}
	if question.FalseCriteria != "" {
		criteria["false"] = question.FalseCriteria
	}
	return criteria
}

// ChoiceCriteria builds the option-to-description map for a Choice
// question; options without a description are sent as null.
func ChoiceCriteria(question fantasy.EvaluationQuestion) map[string]*string {
	criteria := make(map[string]*string, len(question.Options))
	for option, description := range question.Options {
		if description == "" {
			criteria[option] = nil
			continue
		}
		criteria[option] = &description
	}
	return criteria
}

// DecodeResponse converts a TypeSafe wire response. The original questions
// recover rubric labels when the server omits a legend.
func DecodeResponse(wire WireResponse, questions map[string]fantasy.EvaluationQuestion, providerName string) (*fantasy.EvaluationResponse, error) {
	answers := make(map[string]fantasy.EvaluationAnswer, len(wire.Answers))
	for name, wireAnswer := range wire.Answers {
		answer, err := DecodeAnswer(wireAnswer, questions[name])
		if err != nil {
			return nil, fmt.Errorf("answer %q: %w", name, err)
		}
		answers[name] = answer
	}
	resp := &fantasy.EvaluationResponse{
		Model:   wire.Model,
		Answers: answers,
		Usage: fantasy.EvaluationUsage{
			InputTokens:  wire.Usage.InputTokens,
			OutputTokens: wire.Usage.OutputTokens,
			TotalTokens:  wire.Usage.InputTokens + wire.Usage.OutputTokens,
		},
	}
	if wire.LatencyMS != nil {
		resp.ProviderMetadata = fantasy.ProviderMetadata{
			providerName: &ProviderMetadata{LatencyMS: *wire.LatencyMS},
		}
	}
	return resp, nil
}

// DecodeAnswer converts a single wire answer.
func DecodeAnswer(wire WireAnswer, question fantasy.EvaluationQuestion) (fantasy.EvaluationAnswer, error) {
	switch wire.Type {
	case WireTypeNoul:
		if wire.Noul == nil {
			return fantasy.EvaluationAnswer{}, errors.New("noul answer missing probability")
		}
		return fantasy.EvaluationAnswer{Type: fantasy.EvaluationQuestionTypeBool, Probability: *wire.Noul}, nil
	case WireTypeChoice:
		return fantasy.EvaluationAnswer{
			Type:          fantasy.EvaluationQuestionTypeChoice,
			Choice:        wire.Choice,
			Probabilities: wire.Probabilities,
			Confidence:    wire.Confidence,
		}, nil
	case WireTypeScore:
		if wire.Score == nil {
			return fantasy.EvaluationAnswer{}, errors.New("score answer missing score")
		}
		levels, err := evalhttp.LegendToLevels(wire.Legend)
		if err != nil {
			return fantasy.EvaluationAnswer{}, fmt.Errorf("legend: %w", err)
		}
		if levels == nil {
			levels = question.Levels
		}
		probs, err := evalhttp.LevelProbabilitiesFromMap(wire.Probabilities)
		if err != nil {
			return fantasy.EvaluationAnswer{}, fmt.Errorf("probabilities: %w", err)
		}
		return fantasy.EvaluationAnswer{
			Type:               fantasy.EvaluationQuestionTypeScore,
			Score:              *wire.Score,
			Levels:             levels,
			LevelProbabilities: probs,
			Confidence:         wire.Confidence,
		}, nil
	default:
		return fantasy.EvaluationAnswer{}, fmt.Errorf("unknown answer type %q", wire.Type)
	}
}
