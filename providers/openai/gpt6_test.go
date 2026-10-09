package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

func TestGPT6SamplingWithoutReasoning(t *testing.T) {
	t.Parallel()
	for _, id := range []string{"gpt-6-sol", "gpt-6-luna"} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			lm := responsesLanguageModel{modelID: id}
			call := testCall(fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}, &ResponsesProviderOptions{
				ReasoningEffort: new(ReasoningEffortNone), Logprobs: 2,
			})
			call.Temperature = new(0.5)
			call.TopP = new(0.9)
			params, warnings, err := lm.prepareParams(call)
			require.NoError(t, err)
			require.Empty(t, warnings)
			require.Equal(t, "none", string(params.Reasoning.Effort))
			require.True(t, params.Temperature.Valid())
			require.Equal(t, 0.5, params.Temperature.Value)
			require.True(t, params.TopP.Valid())
			require.Equal(t, 0.9, params.TopP.Value)
			require.Equal(t, int64(2), params.TopLogprobs.Value)
		})
	}
}

func TestGPT6ReasoningOmitsLogprobs(t *testing.T) {
	t.Parallel()
	lm := responsesLanguageModel{modelID: "gpt-6-astra"}
	params, warnings, err := lm.prepareParams(testCall(
		fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")},
		&ResponsesProviderOptions{
			ReasoningEffort: new(ReasoningEffortHigh), Logprobs: 2,
			Include: []IncludeType{IncludeMessageOutputTextLogprobs, IncludeReasoningEncryptedContent},
		},
	))
	require.NoError(t, err)
	require.Len(t, warnings, 1)
	require.False(t, params.TopLogprobs.Valid())
	require.Len(t, params.Include, 1)
	require.Equal(t, string(IncludeReasoningEncryptedContent), string(params.Include[0]))
}

func TestGPT6Responses(t *testing.T) {
	t.Parallel()
	for _, id := range []string{
		"gpt-6-astra", "gpt-6-sol", "gpt-6-luna",
		"openai.gpt-6-astra", "openai.gpt-6-sol", "openai.gpt-6-luna",
		"us.openai.gpt-6-astra", "us.openai.gpt-6-sol", "us.openai.gpt-6-luna",
		"global.openai.gpt-6-astra", "global.openai.gpt-6-sol", "global.openai.gpt-6-luna",
	} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			provider, err := New(WithAPIKey("test"), WithUseResponsesAPI())
			require.NoError(t, err)
			model, err := provider.LanguageModel(context.Background(), id)
			require.NoError(t, err)
			lm, ok := model.(responsesLanguageModel)
			require.True(t, ok, "GPT-6 tool use requires the Responses API")
			for _, effort := range []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax} {
				call := testCall(fantasy.Prompt{
					testTextMessage(fantasy.MessageRoleSystem, "Be helpful."),
					testTextMessage(fantasy.MessageRoleUser, "hello"),
				}, &ResponsesProviderOptions{ReasoningEffort: &effort, ReasoningSummary: new("auto")})
				call.Temperature = new(0.5)
				call.TopP = new(0.9)
				params, warnings, err := lm.prepareParams(call)
				require.NoError(t, err)
				require.Equal(t, string(effort), string(params.Reasoning.Effort))
				require.Equal(t, "auto", string(params.Reasoning.Summary))
				require.False(t, params.Temperature.Valid())
				require.False(t, params.TopP.Valid())
				require.Len(t, warnings, 2)
				require.Equal(t, "developer", getResponsesModelConfig(id).systemMessageMode)
			}
		})
	}
}
