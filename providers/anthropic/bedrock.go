package anthropic

import (
	"cmp"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/smithy-go/auth/bearer"
)

// resolveBedrockRegion picks an explicit region, falls back to
// AWS_REGION, then defaults to us-east-1.
func resolveBedrockRegion(region string) string {
	return cmp.Or(region, os.Getenv("AWS_REGION"), "us-east-1")
}

func bedrockBasicAuthConfig(apiKey, region string) aws.Config {
	return aws.Config{
		Region:                  resolveBedrockRegion(region),
		BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
	}
}

// bedrockInferenceProfilePrefixes are the cross-region inference
// profile prefixes Bedrock model IDs may already carry. IDs starting
// with one of these are fully qualified and must not be prefixed
// again with the caller's region.
var bedrockInferenceProfilePrefixes = []string{
	"global.",
	"us-gov.",
	"us.",
	"eu.",
	"apac.",
	"ap.",
	"jp.",
	"au.",
	"ca.",
	"sa.",
	"il.",
	"mx.",
}

func bedrockPrefixModelWithRegion(modelID, region string) string {
	for _, prefix := range bedrockInferenceProfilePrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return modelID
		}
	}
	region = resolveBedrockRegion(region)
	if len(region) < 2 {
		return modelID
	}
	return region[:2] + "." + modelID
}
