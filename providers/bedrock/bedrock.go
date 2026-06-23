// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"cmp"
	"context"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/smithy-go/auth/bearer"
	"github.com/charmbracelet/anthropic-sdk-go/option"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/anthropic"
)

type options struct {
	skipAuth         bool
	apiKey           string
	region           string
	baseURL          string
	userAgent        string
	awsConfig        *aws.Config
	anthropicOptions []anthropic.Option
}

const (
	// Name is the name of the Bedrock provider.
	Name = "bedrock"
)

// Option defines a function that configures Bedrock provider options.
type Option = func(*options)

// provider wraps the Anthropic-backed language-model provider and adds
// embedding support through the Bedrock runtime. Language-model calls
// are delegated to the embedded provider unchanged; embeddings use a
// dedicated bedrockruntime client because Bedrock embedding models
// (Titan, Cohere) are not part of the Anthropic Messages API.
type provider struct {
	fantasy.Provider
	opts options
}

// New creates a new Bedrock provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	lm, err := anthropic.New(
		append(
			o.anthropicOptions,
			anthropic.WithName(Name),
			anthropic.WithBedrock(),
			anthropic.WithSkipAuth(o.skipAuth),
		)...,
	)
	if err != nil {
		return nil, err
	}
	return &provider{Provider: lm, opts: o}, nil
}

// resolveRegion picks an explicit region, falls back to AWS_REGION,
// then defaults to us-east-1, matching the Anthropic Bedrock path.
func resolveRegion(region string) string {
	return cmp.Or(region, os.Getenv("AWS_REGION"), "us-east-1")
}

// awsConfig resolves the AWS config used to build the embedding
// runtime client. An explicit config set via WithAWSConfig wins. A
// bearer API key (or skipAuth) yields a static-token config without
// touching the default credential chain, mirroring the Anthropic
// Bedrock behavior. Otherwise the default credential chain is loaded.
func (o options) resolveAWSConfig(ctx context.Context) (aws.Config, error) {
	if o.awsConfig != nil {
		cfg := *o.awsConfig
		cfg.Region = cmp.Or(o.region, cfg.Region, resolveRegion(""))
		return cfg, nil
	}
	if o.skipAuth || o.apiKey != "" {
		return aws.Config{
			Region:                  resolveRegion(o.region),
			BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: o.apiKey}},
		}, nil
	}
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return aws.Config{}, err
	}
	cfg.Region = cmp.Or(o.region, cfg.Region, resolveRegion(""))
	return cfg, nil
}

// WithAPIKey sets the access token for the Bedrock provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithAPIKey(apiKey))
	}
}

// WithHeaders sets the headers for the Bedrock provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Bedrock provider's
// language-model (Anthropic) path. The embedding runtime client uses
// the HTTP client from the AWS config (see WithAWSConfig).
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHTTPClient(client))
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.userAgent = ua
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithUserAgent(ua))
	}
}

// WithBaseURL sets the base URL for the Bedrock provider. It applies to
// both the language-model path and the embedding runtime client.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithBaseURL(baseURL))
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithRegion sets the AWS region for the Bedrock provider.
func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithBedrockRegion(region))
	}
}

// WithAWSConfig provides an explicit AWS config for the embedding
// runtime client. This is useful when the caller already has a
// configured aws.Config (custom credential provider, HTTP client, or
// endpoint) and for testing against a local endpoint.
func WithAWSConfig(cfg aws.Config) Option {
	return func(o *options) {
		o.awsConfig = &cfg
	}
}
