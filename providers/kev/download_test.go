package kev

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

// fakeHub serves a bundle over the Hugging Face resolve URL layout and
// counts requests so tests can assert on network use.
type fakeHub struct {
	t        *testing.T
	files    map[string][]byte
	requests atomic.Int32
	server   *httptest.Server
}

func newFakeHub(t *testing.T, files map[string][]byte) *fakeHub {
	hub := &fakeHub{t: t, files: files}
	hub.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hub.requests.Add(1)
		// /<owner>/<repo>/resolve/<rev>/<name>
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")
		name := parts[len(parts)-1]
		body, ok := hub.files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if rng := r.Header.Get("Range"); strings.HasPrefix(rng, "bytes=") {
			var start int
			_, _ = fmtSscanf(strings.TrimPrefix(rng, "bytes="), &start)
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[start:])
			return
		}
		_, _ = w.Write(body)
	}))
	t.Cleanup(hub.server.Close)
	return hub
}

func fmtSscanf(s string, start *int) (int, error) {
	s = strings.TrimSuffix(s, "-")
	n := 0
	for _, c := range s {
		n = n*10 + int(c-'0')
	}
	*start = n
	return 1, nil
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// makeBundle builds a tiny consistent bundle and returns files + manifest digest.
func makeBundle(t *testing.T) (map[string][]byte, string) {
	t.Helper()
	model := []byte("GGUF-not-really-a-model-but-bytes")
	head := []byte(`{"format":1,"hidden_size":2,"head_dim":2,"temperature":1,"q_weight":[[1,0],[0,1]],"q_bias":[0,0],"k_weight":[[1,0],[0,1]],"k_bias":[0,0]}`)
	manifest := Manifest{
		Format: 1, Run: "test/kev-tiny", Base: "test", Temperature: 1, HiddenSize: 2, HeadDim: 2,
		Model: "model-f16.gguf", Head: "head.json",
		Files: map[string]ManifestFile{
			"model-f16.gguf": {SHA256: digest(model), Size: int64(len(model))},
			"head.json":      {SHA256: digest(head), Size: int64(len(head))},
		},
	}
	manifestBytes, err := json.Marshal(manifest)
	require.NoError(t, err)
	return map[string][]byte{"model-f16.gguf": model, "head.json": head, manifestName: manifestBytes}, digest(manifestBytes)
}

const testCheckpoint Checkpoint = "kev-tiny"

func newDownloader(t *testing.T, hub *fakeHub, pin string) *Downloader {
	return &Downloader{
		CacheDir:        t.TempDir(),
		Endpoint:        hub.server.URL,
		ManifestDigests: map[Checkpoint]string{testCheckpoint: pin},
	}
}

func TestDownloader_ResolvesAndVerifies(t *testing.T) {
	files, pin := makeBundle(t)
	hub := newFakeHub(t, files)
	d := newDownloader(t, hub, pin)

	bundle, err := d.Resolve(context.Background(), testCheckpoint)
	require.NoError(t, err)
	require.Equal(t, filepath.Join(d.CacheDir, "kev-tiny", "model-f16.gguf"), bundle.ModelPath)
	require.Equal(t, "test/kev-tiny", bundle.Manifest.Run)
	require.Equal(t, int32(3), hub.requests.Load(), "manifest + 2 files")

	// Second resolve is fully offline.
	_, err = d.Resolve(context.Background(), testCheckpoint)
	require.NoError(t, err)
	require.Equal(t, int32(3), hub.requests.Load(), "verified cache must not touch the network")
}

func TestDownloader_RejectsUnpinnedCheckpoint(t *testing.T) {
	files, _ := makeBundle(t)
	hub := newFakeHub(t, files)
	d := &Downloader{CacheDir: t.TempDir(), Endpoint: hub.server.URL}
	_, err := d.Resolve(context.Background(), testCheckpoint)
	require.ErrorIs(t, err, ErrUnknownCheckpoint)
	require.Equal(t, int32(0), hub.requests.Load())
}

func TestDownloader_RejectsUntrustedManifest(t *testing.T) {
	files, pin := makeBundle(t)
	// Server hands out a different manifest than the one we pinned.
	files[manifestName] = append([]byte(nil), files[manifestName]...)
	files[manifestName] = append(files[manifestName], ' ')
	hub := newFakeHub(t, files)
	d := newDownloader(t, hub, pin)

	_, err := d.Resolve(context.Background(), testCheckpoint)
	require.ErrorIs(t, err, ErrManifestUntrusted)
	require.Equal(t, int32(1), hub.requests.Load(), "no files may be fetched after an untrusted manifest")
	_, statErr := os.Stat(filepath.Join(d.CacheDir, "kev-tiny", manifestName))
	require.True(t, os.IsNotExist(statErr), "untrusted manifest must not be left in the cache")
}

func TestDownloader_RejectsCorruptFileAndRefetches(t *testing.T) {
	files, pin := makeBundle(t)
	hub := newFakeHub(t, files)
	d := newDownloader(t, hub, pin)
	_, err := d.Resolve(context.Background(), testCheckpoint)
	require.NoError(t, err)

	// Corrupt the cached model in place; next resolve must detect and refetch.
	modelPath := filepath.Join(d.CacheDir, "kev-tiny", "model-f16.gguf")
	require.NoError(t, os.WriteFile(modelPath, []byte("garbage of same-ish length!!!!!!!"), 0o600))
	before := hub.requests.Load()
	_, err = d.Resolve(context.Background(), testCheckpoint)
	require.NoError(t, err)
	require.Equal(t, before+1, hub.requests.Load(), "only the corrupt file is refetched")
	data, _ := os.ReadFile(modelPath)
	require.Equal(t, files["model-f16.gguf"], data)

	// A server that keeps serving bad bytes yields ErrChecksum.
	hub.files["model-f16.gguf"] = []byte("evil")
	require.NoError(t, os.Remove(modelPath))
	_, err = d.Resolve(context.Background(), testCheckpoint)
	require.ErrorIs(t, err, ErrChecksum)
}

func TestDownloader_ResumesPartial(t *testing.T) {
	files, pin := makeBundle(t)
	hub := newFakeHub(t, files)
	d := newDownloader(t, hub, pin)

	dir := filepath.Join(d.CacheDir, "kev-tiny")
	require.NoError(t, os.MkdirAll(dir, 0o750))
	require.NoError(t, os.WriteFile(filepath.Join(dir, manifestName), files[manifestName], 0o600))
	partial := files["model-f16.gguf"][:10]
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model-f16.gguf"+partSuffix), partial, 0o600))

	_, err := d.Resolve(context.Background(), testCheckpoint)
	require.NoError(t, err)
	data, _ := os.ReadFile(filepath.Join(dir, "model-f16.gguf"))
	require.Equal(t, files["model-f16.gguf"], data)
}

func TestDownloader_LocalBundleDirectory(t *testing.T) {
	files, _ := makeBundle(t)
	dir := t.TempDir()
	for name, data := range files {
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), data, 0o600))
	}
	d := &Downloader{CacheDir: t.TempDir()}
	bundle, err := d.Resolve(context.Background(), Checkpoint(dir))
	require.NoError(t, err)
	require.Equal(t, filepath.Join(dir, "head.json"), bundle.HeadPath)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "head.json"), []byte("{}"), 0o600))
	_, err = d.Resolve(context.Background(), Checkpoint(dir))
	require.ErrorIs(t, err, ErrChecksum)
}

func TestDownloader_HTTPFailure(t *testing.T) {
	_, pin := makeBundle(t)
	hub := newFakeHub(t, map[string][]byte{})
	d := newDownloader(t, hub, pin)
	_, err := d.Resolve(context.Background(), testCheckpoint)
	require.ErrorIs(t, err, ErrDownloadFailed)
}

func TestPinnedCheckpointsAreComplete(t *testing.T) {
	for _, ckpt := range Checkpoints() {
		pin, ok := manifestDigests[ckpt]
		require.True(t, ok, "%s has no pinned manifest digest", ckpt)
		require.Len(t, pin, 64)
	}
}
