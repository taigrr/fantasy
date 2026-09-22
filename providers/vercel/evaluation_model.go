package vercel

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/evalhttp"
	"github.com/taigrr/fantasy/providers/typesafe"
)

const (
	// PathEvaluate is the gateway evaluation endpoint, relative to DefaultURL.
	PathEvaluate = "/evaluate"

	// ModelJev is the gateway slug for TypeSafe's Jev decision model.
	ModelJev = "typesafe-ai/jev"

	// EvaluationProviderTypeSafe is the gateway's provider slug for TypeSafe,
	// for use in EvaluationOptions.Only / Order.
	EvaluationProviderTypeSafe = "typesafe-ai"

	// WireTypeBoolean is the gateway's wire name for Bool questions.
	WireTypeBoolean = "boolean"
	// WireTypeChoice is the wire name for Choice questions.
	WireTypeChoice = "choice"
	// WireTypeScore is the wire name for Score questions.
	WireTypeScore = "score"

	// TypeEvaluationOptions is the global type identifier for per-call
	// evaluation options.
	TypeEvaluationOptions = Name + ".evaluation.options"
	// TypeEvaluationMetadata is the global type identifier for evaluation
	// response metadata.
	TypeEvaluationMetadata = Name + ".evaluation.metadata"
)

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
	fantasy.RegisterProviderType(TypeEvaluationMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v EvaluationMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

type evaluationOptions struct {
	baseURL    string
	apiKey     string
	name       string
	headers    map[string]string
	userAgent  string
	httpClient evalhttp.HTTPDoer
	retry      fantasy.RetryOptions
	defaults   *EvaluationOptions
}

func newEvaluationOptions() evaluationOptions {
	return evaluationOptions{
		baseURL: DefaultURL,
		name:    Name,
		headers: map[string]string{},
		retry:   fantasy.DefaultRetryOptions(),
	}
}

// WithEvaluationRetryOptions configures retries for evaluation calls.
func WithEvaluationRetryOptions(retry fantasy.RetryOptions) Option {
	return func(o *options) { o.evaluation.retry = retry }
}

// WithEvaluationOptions sets default gateway options applied to every
// evaluation call unless the call carries its own.
func WithEvaluationOptions(opts EvaluationOptions) Option {
	return func(o *options) { o.evaluation.defaults = &opts }
}

// EvaluationOptions are gateway routing options for evaluation calls,
// forwarded under providerOptions.gateway.
type EvaluationOptions struct {
	// ZeroDataRetention requires the request be served with ZDR.
	ZeroDataRetention bool `json:"zeroDataRetention,omitempty"`
	// Only restricts which upstream providers may serve the request.
	Only []string `json:"only,omitempty"`
	// Order sets the preferred provider order.
	Order []string `json:"order,omitempty"`
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

// EvaluationRouting is the gateway's routing report.
type EvaluationRouting struct {
	OriginalModelID  string `json:"originalModelId,omitempty"`
	ResolvedProvider string `json:"resolvedProvider,omitempty"`
	CanonicalSlug    string `json:"canonicalSlug,omitempty"`
	FinalProvider    string `json:"finalProvider,omitempty"`
}

// EvaluationMetadata is the gateway's response metadata for evaluations.
type EvaluationMetadata struct {
	Routing       *EvaluationRouting `json:"routing,omitempty"`
	Cost          string             `json:"cost,omitempty"`
	MarketCost    string             `json:"marketCost,omitempty"`
	SurchargeCost string             `json:"surchargeCost,omitempty"`
	GatewayCost   string             `json:"gatewayCost,omitempty"`
	GenerationID  string             `json:"generationId,omitempty"`
}

// Options implements fantasy.ProviderOptionsData.
func (*EvaluationMetadata) Options() {}

// MarshalJSON implements json.Marshaler.
func (m EvaluationMetadata) MarshalJSON() ([]byte, error) {
	type plain EvaluationMetadata
	return fantasy.MarshalProviderType(TypeEvaluationMetadata, plain(m))
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *EvaluationMetadata) UnmarshalJSON(data []byte) error {
	type plain EvaluationMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = EvaluationMetadata(p)
	return nil
}

// EvaluationModel implements fantasy.EvaluationProvider.
func (p *provider) EvaluationModel(_ context.Context, modelID string) (fantasy.EvaluationModel, error) {
	return &evaluationModel{
		provider: p.evaluation.name,
		modelID:  cmp.Or(modelID, ModelJev),
		defaults: p.evaluation.defaults,
		client: &evalhttp.Client{
			BaseURL:    p.evaluation.baseURL,
			APIKey:     p.evaluation.apiKey,
			Headers:    p.evaluation.headers,
			UserAgent:  p.evaluation.userAgent,
			HTTPClient: p.evaluation.httpClient,
			Retry:      p.evaluation.retry,
		},
	}, nil
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
	gateway := m.defaults
	if raw, ok := call.ProviderOptions[m.provider]; ok {
		if opts, ok := raw.(*EvaluationOptions); ok {
			gateway = opts
		}
	}
	wireReq, err := EncodeEvaluationRequest(call, m.modelID, gateway)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.provider, err)
	}
	var wireResp EvaluationWireResponse
	if err := m.client.PostJSON(ctx, evalhttp.Request{Path: PathEvaluate, Body: wireReq, UserAgent: call.UserAgent}, &wireResp); err != nil {
		return nil, err
	}
	resp, err := DecodeEvaluationResponse(wireResp, call.Questions, m.provider)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.provider, err)
	}
	return resp, nil
}

// EvaluationWireQuestion is the on-the-wire question shape.
type EvaluationWireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// EvaluationWireProviderOptions is the providerOptions envelope.
type EvaluationWireProviderOptions struct {
	Gateway *EvaluationOptions `json:"gateway,omitempty"`
}

// MarshalJSON emits the gateway options as a plain object, without the
// fantasy type envelope used for ProviderOptions round-tripping.
func (o EvaluationWireProviderOptions) MarshalJSON() ([]byte, error) {
	type plain EvaluationOptions
	envelope := struct {
		Gateway *plain `json:"gateway,omitempty"`
	}{}
	if o.Gateway != nil {
		converted := plain(*o.Gateway)
		envelope.Gateway = &converted
	}
	return json.Marshal(envelope)
}

// EvaluationWireRequest is the on-the-wire request shape.
type EvaluationWireRequest struct {
	Model           string                            `json:"model"`
	State           json.RawMessage                   `json:"state"`
	Questions       map[string]EvaluationWireQuestion `json:"questions"`
	ProviderOptions *EvaluationWireProviderOptions    `json:"providerOptions,omitempty"`
}

// EvaluationWireAnswer is the on-the-wire answer shape.
type EvaluationWireAnswer struct {
	Type          string             `json:"type"`
	Probability   *float64           `json:"probability,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// EvaluationWireUsage is the on-the-wire usage shape.
type EvaluationWireUsage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
	TotalTokens  int64 `json:"totalTokens,omitempty"`
}

// EvaluationWireResponse is the on-the-wire response shape.
type EvaluationWireResponse struct {
	Model            string                          `json:"model"`
	Answers          map[string]EvaluationWireAnswer `json:"answers"`
	Usage            EvaluationWireUsage             `json:"usage"`
	ProviderMetadata struct {
		Gateway *EvaluationMetadataWire `json:"gateway,omitempty"`
	} `json:"providerMetadata"`
}

// EvaluationMetadataWire is the plain (un-enveloped) gateway metadata.
type EvaluationMetadataWire struct {
	Routing       *EvaluationRouting `json:"routing,omitempty"`
	Cost          string             `json:"cost,omitempty"`
	MarketCost    string             `json:"marketCost,omitempty"`
	SurchargeCost string             `json:"surchargeCost,omitempty"`
	GatewayCost   string             `json:"gatewayCost,omitempty"`
	GenerationID  string             `json:"generationId,omitempty"`
}

// EncodeEvaluationRequest converts an EvaluationCall into the gateway wire format.
func EncodeEvaluationRequest(call fantasy.EvaluationCall, model string, gateway *EvaluationOptions) (EvaluationWireRequest, error) {
	state, err := json.Marshal(call.State)
	if err != nil {
		return EvaluationWireRequest{}, fmt.Errorf("marshal state: %w", err)
	}
	questions := make(map[string]EvaluationWireQuestion, len(call.Questions))
	for name, question := range call.Questions {
		wireQuestion, err := EncodeEvaluationQuestion(question)
		if err != nil {
			return EvaluationWireRequest{}, fmt.Errorf("question %q: %w", name, err)
		}
		questions[name] = wireQuestion
	}
	wire := EvaluationWireRequest{Model: model, State: state, Questions: questions}
	if gateway != nil {
		wire.ProviderOptions = &EvaluationWireProviderOptions{Gateway: gateway}
	}
	return wire, nil
}

// EncodeEvaluationQuestion converts a single question into the gateway wire format.
func EncodeEvaluationQuestion(question fantasy.EvaluationQuestion) (EvaluationWireQuestion, error) {
	switch question.Type {
	case fantasy.EvaluationQuestionTypeBool:
		wire := EvaluationWireQuestion{Type: WireTypeBoolean, Instructions: question.Instructions}
		if criteria := typesafe.BoolCriteria(question); len(criteria) > 0 {
			wire.Criteria = criteria
		}
		return wire, nil
	case fantasy.EvaluationQuestionTypeChoice:
		return EvaluationWireQuestion{Type: WireTypeChoice, Instructions: question.Instructions, Criteria: typesafe.ChoiceCriteria(question)}, nil
	case fantasy.EvaluationQuestionTypeScore:
		return EvaluationWireQuestion{Type: WireTypeScore, Instructions: question.Instructions, Criteria: question.Levels}, nil
	default:
		return EvaluationWireQuestion{}, fmt.Errorf("unknown question type %q", question.Type)
	}
}

// DecodeEvaluationResponse converts a gateway wire response.
func DecodeEvaluationResponse(wire EvaluationWireResponse, questions map[string]fantasy.EvaluationQuestion, providerName string) (*fantasy.EvaluationResponse, error) {
	answers := make(map[string]fantasy.EvaluationAnswer, len(wire.Answers))
	for name, wireAnswer := range wire.Answers {
		answer, err := DecodeEvaluationAnswer(wireAnswer, questions[name])
		if err != nil {
			return nil, fmt.Errorf("answer %q: %w", name, err)
		}
		answers[name] = answer
	}
	total := wire.Usage.TotalTokens
	if total == 0 {
		total = wire.Usage.InputTokens + wire.Usage.OutputTokens
	}
	resp := &fantasy.EvaluationResponse{
		Model:   wire.Model,
		Answers: answers,
		Usage: fantasy.EvaluationUsage{
			InputTokens:  wire.Usage.InputTokens,
			OutputTokens: wire.Usage.OutputTokens,
			TotalTokens:  total,
		},
	}
	if gw := wire.ProviderMetadata.Gateway; gw != nil {
		resp.ProviderMetadata = fantasy.ProviderMetadata{
			providerName: &EvaluationMetadata{
				Routing:       gw.Routing,
				Cost:          gw.Cost,
				MarketCost:    gw.MarketCost,
				SurchargeCost: gw.SurchargeCost,
				GatewayCost:   gw.GatewayCost,
				GenerationID:  gw.GenerationID,
			},
		}
	}
	return resp, nil
}

// DecodeEvaluationAnswer converts a single wire answer.
func DecodeEvaluationAnswer(wire EvaluationWireAnswer, question fantasy.EvaluationQuestion) (fantasy.EvaluationAnswer, error) {
	switch wire.Type {
	case WireTypeBoolean:
		if wire.Probability == nil {
			return fantasy.EvaluationAnswer{}, errors.New("boolean answer missing probability")
		}
		return fantasy.EvaluationAnswer{Type: fantasy.EvaluationQuestionTypeBool, Probability: *wire.Probability}, nil
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
