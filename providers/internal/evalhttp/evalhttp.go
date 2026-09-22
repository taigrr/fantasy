// Package evalhttp is the shared JSON-over-HTTP layer for evaluation
// providers. It maps failures onto fantasy.ProviderError and retries with
// exponential backoff while honouring retry-after headers.
package evalhttp

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/taigrr/fantasy"
)

const (
	headerAuthorization = "Authorization"
	headerContentType   = "Content-Type"
	headerUserAgent     = "User-Agent"
	contentTypeJSON     = "application/json"
	bearerPrefix        = "Bearer "

	maxErrorBody = 64 << 10
	maxBody      = maxErrorBody * 16
)

// HTTPDoer is the subset of *http.Client the transport needs. It matches the
// OpenAI SDK's option.HTTPClient so one client can serve both modalities.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// Client posts JSON to a base URL with bearer auth and retries.
type Client struct {
	BaseURL    string
	APIKey     string
	Headers    map[string]string
	UserAgent  string
	HTTPClient HTTPDoer
	Retry      fantasy.RetryOptions
}

// Request carries per-call overrides.
type Request struct {
	Path      string
	Body      any
	UserAgent string
}

// PostJSON marshals req.Body, posts it and decodes the response into out.
func (c *Client) PostJSON(ctx context.Context, req Request, out any) error {
	body, err := json.Marshal(req.Body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	return c.roundTrip(ctx, http.MethodPost, req.Path, body, req.UserAgent, out)
}

// GetJSON fetches path and decodes the response into out.
func (c *Client) GetJSON(ctx context.Context, path string, out any) error {
	return c.roundTrip(ctx, http.MethodGet, path, nil, "", out)
}

func (c *Client) roundTrip(ctx context.Context, method, path string, body []byte, userAgent string, out any) error {
	retry := fantasy.RetryWithExponentialBackoffRespectingRetryHeaders[[]byte](c.Retry)
	respBody, err := retry(ctx, func() ([]byte, error) {
		return c.do(ctx, method, path, body, userAgent)
	})
	if err != nil {
		return err
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func (c *Client) do(ctx context.Context, method, path string, body []byte, userAgent string) ([]byte, error) {
	url := strings.TrimRight(c.BaseURL, "/") + "/" + strings.TrimLeft(path, "/")
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set(headerContentType, contentTypeJSON)
	}
	if ua := cmp.Or(userAgent, c.UserAgent); ua != "" {
		req.Header.Set(headerUserAgent, ua)
	}
	if c.APIKey != "" {
		req.Header.Set(headerAuthorization, bearerPrefix+c.APIKey)
	}
	for key, value := range c.Headers {
		req.Header.Set(key, value)
	}

	var httpClient HTTPDoer = http.DefaultClient
	if c.HTTPClient != nil {
		httpClient = c.HTTPClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		if fantasy.IsTransportError(err) {
			return nil, fantasy.WrapTransportError(err)
		}
		return nil, &fantasy.ProviderError{
			Title:       "request failed",
			Message:     err.Error(),
			Cause:       err,
			URL:         url,
			RequestBody: body,
		}
	}
	defer resp.Body.Close() //nolint:errcheck

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return nil, &fantasy.ProviderError{
			Title:           "read response failed",
			Message:         err.Error(),
			Cause:           err,
			URL:             url,
			StatusCode:      resp.StatusCode,
			RequestBody:     body,
			ResponseHeaders: headerMap(resp.Header),
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if len(respBody) > maxErrorBody {
			respBody = respBody[:maxErrorBody]
		}
		return nil, &fantasy.ProviderError{
			Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(resp.StatusCode), "provider request failed"),
			Message:         errorMessage(respBody, resp.Status),
			URL:             url,
			StatusCode:      resp.StatusCode,
			RequestBody:     body,
			ResponseHeaders: headerMap(resp.Header),
			ResponseBody:    respBody,
		}
	}
	return respBody, nil
}

func errorMessage(body []byte, status string) string {
	var envelope struct {
		Error   any    `json:"error"`
		Message string `json:"message"`
		Detail  any    `json:"detail"`
	}
	if err := json.Unmarshal(body, &envelope); err == nil {
		switch errVal := envelope.Error.(type) {
		case string:
			if errVal != "" {
				return errVal
			}
		case map[string]any:
			if msg, ok := errVal["message"].(string); ok && msg != "" {
				return msg
			}
		}
		if envelope.Message != "" {
			return envelope.Message
		}
		if envelope.Detail != nil {
			if detail, err := json.Marshal(envelope.Detail); err == nil {
				return string(detail)
			}
		}
	}
	if trimmed := strings.TrimSpace(string(body)); trimmed != "" {
		return trimmed
	}
	return status
}

func headerMap(header http.Header) map[string]string {
	out := make(map[string]string, len(header))
	for key, values := range header {
		out[strings.ToLower(key)] = strings.Join(values, ", ")
	}
	return out
}
