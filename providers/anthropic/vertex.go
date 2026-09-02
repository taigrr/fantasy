//go:build fantasy_google

package anthropic

import (
	"context"

	"github.com/charmbracelet/anthropic-sdk-go/option"
	"github.com/charmbracelet/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// VertexEnabled reports whether Vertex AI support was compiled in. It is
// true only when built with the fantasy_google build tag.
const VertexEnabled = true

type googleDummyTokenSource struct{}

func (googleDummyTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: "dummy-token"}, nil
}

// vertexRequestOptions returns the request options that route the client
// through Vertex AI using application-default credentials.
func vertexRequestOptions(ctx context.Context, o options) ([]option.RequestOption, error) {
	var credentials *google.Credentials
	if o.skipAuth {
		credentials = &google.Credentials{TokenSource: &googleDummyTokenSource{}}
	} else {
		var err error
		credentials, err = google.FindDefaultCredentials(ctx, VertexAuthScope)
		if err != nil {
			return nil, err
		}
	}
	return []option.RequestOption{
		vertex.WithCredentials(ctx, o.vertexLocation, o.vertexProject, credentials),
	}, nil
}
