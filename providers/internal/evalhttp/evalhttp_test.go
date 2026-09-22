package evalhttp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/taigrr/fantasy"
)

func TestPostJSONRetriesOn429ThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"slow down"}`))
			return
		}
		if r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		if r.Header.Get("X-Extra") != "1" {
			t.Errorf("missing extra header")
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	client := &Client{
		BaseURL: server.URL + "/",
		APIKey:  "key",
		Headers: map[string]string{"X-Extra": "1"},
		Retry:   fantasy.RetryOptions{MaxRetries: 2, InitialDelayIn: time.Millisecond, BackoffFactor: 1},
	}
	var out struct {
		OK bool `json:"ok"`
	}
	if err := client.PostJSON(context.Background(), Request{Path: "/v1/x", Body: map[string]string{"a": "b"}}, &out); err != nil {
		t.Fatal(err)
	}
	if !out.OK || calls.Load() != 2 {
		t.Fatalf("ok=%v calls=%d", out.OK, calls.Load())
	}
}

func TestPostJSONNonRetryableError(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":[{"loc":["questions"],"msg":"bad"}]}`))
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, Retry: fantasy.RetryOptions{MaxRetries: 3, InitialDelayIn: time.Millisecond, BackoffFactor: 1}}
	err := client.PostJSON(context.Background(), Request{Path: "v1/x", Body: map[string]string{}}, nil)
	var providerErr *fantasy.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %v", err, err)
	}
	if providerErr.StatusCode != http.StatusUnprocessableEntity || providerErr.IsRetryable() {
		t.Fatalf("unexpected error %+v", providerErr)
	}
	if providerErr.Message == "" || providerErr.Message == "422 Unprocessable Entity" {
		t.Fatalf("expected detail-derived message, got %q", providerErr.Message)
	}
	if calls.Load() != 1 {
		t.Fatalf("calls = %d", calls.Load())
	}
}

func TestErrorMessageShapes(t *testing.T) {
	tests := map[string]string{
		`{"error":"plain"}`:              "plain",
		`{"error":{"message":"nested"}}`: "nested",
		`{"message":"top"}`:              "top",
		`not json`:                       "not json",
		``:                               "500 Internal Server Error",
		`{"detail":"string detail"}`:     `"string detail"`,
		`{"error":{"code":"x"},"message":"fallback"}`: "fallback",
	}
	for body, want := range tests {
		if got := errorMessage([]byte(body), "500 Internal Server Error"); got != want {
			t.Errorf("errorMessage(%q) = %q, want %q", body, got, want)
		}
	}
}

func TestGetJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.Header.Get("Content-Type") != "" {
			t.Errorf("unexpected method/content-type: %s %q", r.Method, r.Header.Get("Content-Type"))
		}
		_, _ = w.Write([]byte(`{"n":1}`))
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL}
	var out struct {
		N int `json:"n"`
	}
	if err := client.GetJSON(context.Background(), "/v1/models", &out); err != nil || out.N != 1 {
		t.Fatalf("out=%+v err=%v", out, err)
	}
}

func TestContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	client := &Client{BaseURL: server.URL, Retry: fantasy.DefaultRetryOptions()}
	err := client.PostJSON(ctx, Request{Path: "/x", Body: map[string]string{}}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}
