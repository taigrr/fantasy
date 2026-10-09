package clef

import (
	"context"
	"net/http"

	"github.com/taigrr/fantasy/providers/internal/localgguf"
)

// Checkpoint names a published Clef bundle: one GGUF holding the merged
// backbone and the joint schema head as a llama.cpp decision model, plus a
// manifest with checksums.
type Checkpoint string

// Published checkpoints.
const (
	// CheckpointFlash is Clef-Flash, the 9B model, stored Q8_0 (9.7 GB).
	CheckpointFlash Checkpoint = "clef-flash"

	// DefaultCheckpoint is the one that fits a workstation.
	DefaultCheckpoint = CheckpointFlash

	// DefaultRepoOwner is the Hugging Face account hosting the GGUF bundles.
	DefaultRepoOwner = localgguf.DefaultRepoOwner
	// EnvCacheDir overrides the download location.
	EnvCacheDir = "CLEF_CACHE"
	// EnvHFEndpoint overrides the Hugging Face host (mirrors, offline caches).
	EnvHFEndpoint = localgguf.EnvHFEndpoint
	// EnvHFToken supplies a token for private repos or higher rate limits.
	EnvHFToken = localgguf.EnvHFToken

	cacheSubdir = "clef"
)

// Sentinel errors for checkpoint downloads.
var (
	// ErrChecksum means a bundle file does not match its manifest.
	ErrChecksum = localgguf.ErrChecksum
	// ErrManifestUntrusted means a bundle manifest does not hash to the
	// digest compiled into this package for that checkpoint.
	ErrManifestUntrusted = localgguf.ErrManifestUntrusted
	// ErrUnknownCheckpoint means the checkpoint is neither a published name
	// with a pinned manifest digest nor a local bundle directory.
	ErrUnknownCheckpoint = localgguf.ErrUnknownCheckpoint
	// ErrDownloadFailed wraps HTTP and I/O failures while fetching a bundle.
	ErrDownloadFailed = localgguf.ErrDownloadFailed
)

// manifestDigests pins the SHA-256 of manifest.json for each published
// checkpoint. Regenerate with gojev/tools/clef-convert after re-converting.
var manifestDigests = map[Checkpoint]string{
	CheckpointFlash: "",
}

// Checkpoints lists the published checkpoints this build knows how to verify.
func Checkpoints() []Checkpoint {
	return []Checkpoint{CheckpointFlash}
}

// Manifest describes a bundle. It is written by the conversion script.
type Manifest = localgguf.Manifest

// ManifestFile is one entry in Manifest.Files.
type ManifestFile = localgguf.ManifestFile

// Bundle is a resolved, verified checkpoint on local disk.
type Bundle = localgguf.Bundle

// Progress reports download progress for one file.
type Progress = localgguf.Progress

// Downloader fetches checkpoint bundles into a local cache.
type Downloader struct {
	// CacheDir defaults to $CLEF_CACHE, then $XDG_CACHE_HOME/clef or ~/.cache/clef.
	CacheDir string
	// RepoOwner defaults to DefaultRepoOwner; the repo is <owner>/<checkpoint>-gguf.
	RepoOwner string
	// Revision defaults to "main".
	Revision string
	// Endpoint defaults to $HF_ENDPOINT or https://huggingface.co.
	Endpoint string
	// ManifestDigests overrides the compiled-in manifest pins, for testing
	// or for serving privately converted checkpoints under known digests.
	ManifestDigests map[Checkpoint]string
	// Token defaults to $HF_TOKEN.
	Token string
	// HTTPClient defaults to a client with no overall timeout (files are large).
	HTTPClient *http.Client
	// Progress is optional.
	Progress Progress
}

// DefaultCacheDir resolves the cache directory from the environment.
func DefaultCacheDir() (string, error) {
	return localgguf.DefaultCacheDir(EnvCacheDir, cacheSubdir)
}

func (d *Downloader) cacheDir() (string, error) {
	if d.CacheDir != "" {
		return d.CacheDir, nil
	}
	return DefaultCacheDir()
}

func (d *Downloader) manifestDigest(ckpt Checkpoint) (string, bool) {
	if d.ManifestDigests != nil {
		if digest, ok := d.ManifestDigests[ckpt]; ok {
			return digest, true
		}
	}
	digest, ok := manifestDigests[ckpt]
	return digest, ok && digest != ""
}

func (d *Downloader) shared(ckpt Checkpoint) (*localgguf.Downloader, error) {
	cache, err := d.cacheDir()
	if err != nil {
		return nil, err
	}
	digests := map[string]string{}
	if digest, ok := d.manifestDigest(ckpt); ok {
		digests[string(ckpt)] = digest
	}
	return &localgguf.Downloader{
		CacheDir:        cache,
		RepoOwner:       d.RepoOwner,
		Revision:        d.Revision,
		Endpoint:        d.Endpoint,
		ManifestDigests: digests,
		Token:           d.Token,
		HTTPClient:      d.HTTPClient,
		Progress:        d.Progress,
	}, nil
}

// Resolve returns a verified local bundle, downloading anything missing or
// corrupt. Local bundle directories (containing manifest.json) may be passed
// directly as the checkpoint to skip the network entirely.
func (d *Downloader) Resolve(ctx context.Context, ckpt Checkpoint) (*Bundle, error) {
	shared, err := d.shared(ckpt)
	if err != nil {
		return nil, err
	}
	return shared.Resolve(ctx, string(ckpt))
}
