//go:build !fantasy_google

package google

import (
	"context"

	"github.com/taigrr/fantasy"
)

// Enabled reports whether the Gemini/Vertex client was compiled in.
const Enabled = false

// LanguageModel implements fantasy.Provider. It always fails because the
// Gemini client was not compiled in; build with -tags fantasy_google.
func (a *provider) LanguageModel(context.Context, string) (fantasy.LanguageModel, error) {
	return nil, ErrNotCompiled
}
