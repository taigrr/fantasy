package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/taigrr/fantasy"
)

func TestThinkingDisplay(t *testing.T) {
	t.Parallel()
	for _, effort := range []*Effort{nil, new(EffortMedium), new(EffortMax)} {
		opts, err := ParseOptions(map[string]any{"thinking_display": "summarized"})
		require.NoError(t, err)
		opts.Effort = effort
		lm := languageModel{modelID: "claude-opus-5-5"}
		params, _, warnings, _, err := lm.prepareParams(fantasy.Call{
			Prompt: testPrompt(), ProviderOptions: NewProviderOptions(opts),
		})
		require.NoError(t, err)
		require.Empty(t, warnings)
		data, err := json.Marshal(params)
		require.NoError(t, err)
		var body map[string]any
		require.NoError(t, json.Unmarshal(data, &body))
		require.Equal(t, map[string]any{"type": "adaptive", "display": "summarized"}, body["thinking"])
		if effort != nil {
			require.Equal(t, map[string]any{"effort": string(*effort)}, body["output_config"])
		} else {
			require.NotContains(t, body, "output_config")
		}
	}
}
