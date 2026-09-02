package anthropic

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBedrockPrefixModelWithRegion(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		modelID string
		region  string
		want    string
	}{
		{
			name:    "bare model id gets region prefix",
			modelID: "anthropic.claude-fable-5-1",
			region:  "us-east-1",
			want:    "us.anthropic.claude-fable-5-1",
		},
		{
			name:    "matching geo prefix is preserved",
			modelID: "us.anthropic.claude-fable-5-1",
			region:  "us-east-1",
			want:    "us.anthropic.claude-fable-5-1",
		},
		{
			name:    "global inference profile is preserved",
			modelID: "global.anthropic.claude-fable-5-1",
			region:  "us-east-1",
			want:    "global.anthropic.claude-fable-5-1",
		},
		{
			name:    "global inference profile is preserved outside us",
			modelID: "global.anthropic.claude-fable-5-1",
			region:  "eu-west-1",
			want:    "global.anthropic.claude-fable-5-1",
		},
		{
			name:    "apac profile is not double prefixed",
			modelID: "apac.anthropic.claude-fable-5-1",
			region:  "ap-southeast-2",
			want:    "apac.anthropic.claude-fable-5-1",
		},
		{
			name:    "cross geo profile is preserved",
			modelID: "eu.anthropic.claude-fable-5-1",
			region:  "us-east-1",
			want:    "eu.anthropic.claude-fable-5-1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, test.want, bedrockPrefixModelWithRegion(test.modelID, test.region))
		})
	}
}
