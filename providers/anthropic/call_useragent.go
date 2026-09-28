package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/httpheaders"
)

func callUARequestOptions(call fantasy.Call) []option.RequestOption {
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}

func callHeadersRequestOptions(call fantasy.Call) []option.RequestOption {
	headers, ok := httpheaders.CallHeaders(call.Headers)
	if !ok {
		return nil
	}
	opts := make([]option.RequestOption, 0, len(headers))
	for k, v := range headers {
		opts = append(opts, option.WithHeader(k, v))
	}
	return opts
}
