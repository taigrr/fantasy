package anthropic

import (
	"github.com/charmbracelet/anthropic-sdk-go/option"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/httpheaders"
)

func callUARequestOptions(call fantasy.Call) []option.RequestOption {
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}
