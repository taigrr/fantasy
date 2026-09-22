package fantasy

import (
	"context"
	"errors"
	"fmt"
)

// EvaluationQuestionType identifies one of the decision-model question
// primitives. Decision models (TypeSafe's Jev, the open-weight Kev) do not
// generate text: they take a state plus typed questions and return one
// calibrated, typed answer per question in a single forward pass.
type EvaluationQuestionType string

const (
	// EvaluationQuestionTypeBool is a yes/no question answered with the
	// probability of "yes". TypeSafe calls this "noul" on the wire and Vercel
	// AI Gateway calls it "boolean"; providers translate.
	EvaluationQuestionTypeBool EvaluationQuestionType = "bool"
	// EvaluationQuestionTypeChoice picks one option from a labelled set.
	EvaluationQuestionTypeChoice EvaluationQuestionType = "choice"
	// EvaluationQuestionTypeScore rates the state on an ordered rubric.
	EvaluationQuestionTypeScore EvaluationQuestionType = "score"
)

const (
	// EvaluationMaxChoiceOptions is the maximum option count for a Choice question.
	EvaluationMaxChoiceOptions = 255
	// EvaluationMinScoreLevels is the minimum rubric size for a Score question.
	EvaluationMinScoreLevels = 2
	// EvaluationMaxScoreLevels is the maximum rubric size for a Score question.
	EvaluationMaxScoreLevels = 10
)

// EvaluationQuestion is a single typed question evaluated against the call
// state. Use BoolQuestion, ChoiceQuestion and ScoreQuestion to build one.
type EvaluationQuestion struct {
	Type         EvaluationQuestionType `json:"type"`
	Instructions string                 `json:"instructions"`

	// TrueCriteria and FalseCriteria optionally describe what counts as a
	// yes or no answer for Bool questions.
	TrueCriteria  string `json:"true_criteria,omitempty"`
	FalseCriteria string `json:"false_criteria,omitempty"`

	// Options maps option names to optional descriptions for Choice questions.
	Options map[string]string `json:"options,omitempty"`

	// Levels is the ordered rubric for Score questions, lowest to highest.
	Levels []string `json:"levels,omitempty"`
}

// BoolQuestion builds a yes/no question.
func BoolQuestion(instructions string) EvaluationQuestion {
	return EvaluationQuestion{Type: EvaluationQuestionTypeBool, Instructions: instructions}
}

// BoolQuestionWithCriteria builds a yes/no question with descriptions of
// what counts as a yes and a no answer. Either description may be empty.
func BoolQuestionWithCriteria(instructions, trueCriteria, falseCriteria string) EvaluationQuestion {
	return EvaluationQuestion{
		Type:          EvaluationQuestionTypeBool,
		Instructions:  instructions,
		TrueCriteria:  trueCriteria,
		FalseCriteria: falseCriteria,
	}
}

// ChoiceQuestion builds a question that picks one option from a labelled set.
func ChoiceQuestion(instructions string, options map[string]string) EvaluationQuestion {
	return EvaluationQuestion{
		Type:         EvaluationQuestionTypeChoice,
		Instructions: instructions,
		Options:      options,
	}
}

// ScoreQuestion builds a question that rates the state on an ordered rubric.
func ScoreQuestion(instructions string, levels ...string) EvaluationQuestion {
	return EvaluationQuestion{
		Type:         EvaluationQuestionTypeScore,
		Instructions: instructions,
		Levels:       levels,
	}
}

// Validate checks that the question is well-formed for its type.
func (q EvaluationQuestion) Validate() error {
	if q.Instructions == "" {
		return errors.New("instructions are required")
	}
	switch q.Type {
	case EvaluationQuestionTypeBool:
		return nil
	case EvaluationQuestionTypeChoice:
		if len(q.Options) == 0 {
			return errors.New("choice question requires at least one option")
		}
		if len(q.Options) > EvaluationMaxChoiceOptions {
			return fmt.Errorf("choice question has %d options, maximum is %d", len(q.Options), EvaluationMaxChoiceOptions)
		}
		return nil
	case EvaluationQuestionTypeScore:
		if len(q.Levels) < EvaluationMinScoreLevels || len(q.Levels) > EvaluationMaxScoreLevels {
			return fmt.Errorf("score question has %d levels, must be between %d and %d", len(q.Levels), EvaluationMinScoreLevels, EvaluationMaxScoreLevels)
		}
		return nil
	default:
		return fmt.Errorf("unknown question type %q", q.Type)
	}
}

// EvaluationCall represents a call to an evaluation (decision) model.
type EvaluationCall struct {
	// State is the content to evaluate: a string, or any JSON-marshalable
	// value (map, slice, struct) for structured input such as a transcript.
	State any `json:"state"`

	// Questions are evaluated in parallel and in isolation against State.
	// Answers come back under the same keys.
	Questions map[string]EvaluationQuestion `json:"questions"`

	// UserAgent overrides the provider-level User-Agent header for this call.
	UserAgent string `json:"-"`

	// ProviderOptions holds provider-specific options, keyed by provider id.
	ProviderOptions ProviderOptions `json:"provider_options"`
}

// Validate checks that the call is well-formed.
func (c EvaluationCall) Validate() error {
	if c.State == nil {
		return errors.New("state is required")
	}
	if len(c.Questions) == 0 {
		return errors.New("at least one question is required")
	}
	for name, question := range c.Questions {
		if err := question.Validate(); err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
	}
	return nil
}

// EvaluationAnswer is the typed result for one question.
type EvaluationAnswer struct {
	Type EvaluationQuestionType `json:"type"`

	// Probability is the probability of "yes" for Bool answers.
	Probability float64 `json:"probability,omitempty"`

	// Choice is the selected option for Choice answers.
	Choice string `json:"choice,omitempty"`
	// Probabilities maps each option to its probability for Choice answers.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`

	// Score is the probability-weighted mean level for Score answers,
	// ranging from 0 to len(Levels)-1.
	Score float64 `json:"score,omitempty"`
	// Levels is the rubric echoed back for Score answers, lowest to highest.
	Levels []string `json:"levels,omitempty"`
	// LevelProbabilities holds the probability of each rubric level, indexed
	// the same as Levels.
	LevelProbabilities []float64 `json:"level_probabilities,omitempty"`

	// Confidence is the model's calibrated confidence in Choice and Score
	// answers when the provider reports one.
	Confidence *float64 `json:"confidence,omitempty"`
}

// Yes reports whether a Bool answer is more likely yes than no.
func (a EvaluationAnswer) Yes() bool {
	return a.Probability >= 0.5
}

// Level returns the rubric label nearest to the score for Score answers.
func (a EvaluationAnswer) Level() string {
	if len(a.Levels) == 0 {
		return ""
	}
	idx := min(max(int(a.Score+0.5), 0), len(a.Levels)-1)
	return a.Levels[idx]
}

// Margin returns the gap between the top two option probabilities for
// Choice answers. A small margin signals an ambiguous decision even when
// the top probability is high in absolute terms. With a single option the
// margin is that option's probability.
func (a EvaluationAnswer) Margin() float64 {
	first, second := 0.0, 0.0
	for _, p := range a.Probabilities {
		if p > first {
			first, second = p, first
		} else if p > second {
			second = p
		}
	}
	return first - second
}

// EvaluationUsage represents token usage for an evaluation call. Decision
// models typically bill input tokens only.
type EvaluationUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// EvaluationResponse represents a response from an evaluation model.
type EvaluationResponse struct {
	// Model is the model that actually served the request, which may be a
	// resolved version of a requested alias.
	Model string `json:"model"`

	// Answers holds one answer per question, keyed as in the call.
	Answers map[string]EvaluationAnswer `json:"answers"`

	Usage    EvaluationUsage `json:"usage"`
	Warnings []CallWarning   `json:"warnings"`

	// ProviderMetadata holds provider specific response metadata, keyed by
	// provider id.
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// EvaluationModel represents a decision model that answers typed questions
// about a state with calibrated probabilities.
type EvaluationModel interface {
	// Evaluate answers the call's questions against its state.
	Evaluate(context.Context, EvaluationCall) (*EvaluationResponse, error)

	// Provider returns the provider id for this model.
	Provider() string
	// Model returns the model id.
	Model() string
}

// EvaluationProvider is an optional capability interface implemented by
// providers that can construct evaluation models. Detect support with a
// type assertion on a [Provider]:
//
//	if ep, ok := p.(fantasy.EvaluationProvider); ok {
//	    em, err := ep.EvaluationModel(ctx, "typesafe-ai/jev")
//	    // ...
//	}
type EvaluationProvider interface {
	Provider
	EvaluationModel(ctx context.Context, modelID string) (EvaluationModel, error)
}
