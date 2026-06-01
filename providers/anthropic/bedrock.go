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

func bedrockPrefixModelWithRegion(modelID, region string) string {
	region = resolveBedrockRegion(region)
	if len(region) < 2 {
		return modelID
	}
	prefix := region[:2] + "."
	if strings.HasPrefix(modelID, prefix) {
		return modelID
	}
	return prefix + modelID
}
