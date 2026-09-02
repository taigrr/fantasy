//go:build !fantasy_google

package anthropic

import (
	"context"

	"github.com/charmbracelet/anthropic-sdk-go/option"
)

// VertexEnabled reports whether Vertex AI support was compiled in. It is
// true only when built with the fantasy_google build tag.
const VertexEnabled = false

// vertexRequestOptions fails because Vertex AI support was not compiled in.
// Build with -tags fantasy_google to enable it.
func vertexRequestOptions(context.Context, options) ([]option.RequestOption, error) {
	return nil, ErrVertexNotCompiled
}
