package kev

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Checkpoint names a published Kev bundle: a GGUF with the LoRA merged in,
// a head.json with the pointer head, and a manifest with checksums.
type Checkpoint string

// Published checkpoints. Larger is more accurate and better calibrated;
// 0.8B is the only one that is quick on a laptop CPU.
const (
	Checkpoint0_8B Checkpoint = "kev-0.8b"
	Checkpoint4B   Checkpoint = "kev-4b"
	Checkpoint9B   Checkpoint = "kev-9b"

	// DefaultCheckpoint balances speed and accuracy for local use.
	DefaultCheckpoint = Checkpoint4B

	// DefaultRepoOwner is the Hugging Face account hosting the GGUF bundles.
	DefaultRepoOwner = "taigrr"
	// EnvCacheDir overrides the download location.
	EnvCacheDir = "KEV_CACHE"
	// EnvHFEndpoint overrides the Hugging Face host (mirrors, offline caches).
	EnvHFEndpoint = "HF_ENDPOINT"
	// EnvHFToken supplies a token for private repos or higher rate limits.
	EnvHFToken = "HF_TOKEN"

	defaultHFEndpoint = "https://huggingface.co"
	manifestName      = "manifest.json"
	partSuffix        = ".part"
	defaultRevision   = "main"
)

// Sentinel errors for checkpoint downloads.
var (
	// ErrChecksum means a bundle file does not match its manifest.
	ErrChecksum = errors.New("kev: checksum mismatch")
	// ErrManifestUntrusted means a bundle manifest does not hash to the
	// digest compiled into this package for that checkpoint. Nothing from
	// it is used.
	ErrManifestUntrusted = errors.New("kev: checkpoint manifest does not match pinned digest")
	// ErrUnknownCheckpoint means the checkpoint is neither a published name
	// with a pinned manifest digest nor a local bundle directory.
	ErrUnknownCheckpoint = errors.New("kev: unknown checkpoint")
	// ErrDownloadFailed wraps HTTP and I/O failures while fetching a bundle.
	ErrDownloadFailed = errors.New("kev: checkpoint download failed")
)

// manifestDigests pins the SHA-256 of manifest.json for each published
// checkpoint, so the module itself is the trust root for the weights: a
// manifest served from Hugging Face is only used if it hashes to the value
// here, and every file it lists is then verified against it.
//
// Regenerate with tools/kev-convert after re-converting a checkpoint.
var manifestDigests = map[Checkpoint]string{
	Checkpoint0_8B: "54e0407e5538c27d269d3d8767cc1907cd0302c6036e331fe982a795124c3277",
	Checkpoint4B:   "6e202eeafa5260b6cbcbdf3ff318cfd15dee11ea7058e304f107b5c338fb21bc",
	Checkpoint9B:   "9b28675b631ad9631fceb2321d16bcc83a147683cfb7d6722e96e5a983d8dd17",
}

// Checkpoints lists the published checkpoints this build knows how to verify.
func Checkpoints() []Checkpoint {
	return []Checkpoint{Checkpoint0_8B, Checkpoint4B, Checkpoint9B}
}

// Manifest describes a bundle. It is written by the conversion script.
type Manifest struct {
	Format      int                     `json:"format"`
	Run         string                  `json:"run"`
	Base        string                  `json:"base"`
	Temperature float64                 `json:"temperature"`
	HiddenSize  int                     `json:"hidden_size"`
	HeadDim     int                     `json:"head_dim"`
	Model       string                  `json:"model"`
	Head        string                  `json:"head"`
	Files       map[string]ManifestFile `json:"files"`
}

// ManifestFile is one entry in Manifest.Files.
type ManifestFile struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// Bundle is a resolved, verified checkpoint on local disk.
type Bundle struct {
	Checkpoint Checkpoint
	Dir        string
	ModelPath  string
	HeadPath   string
	Manifest   Manifest
}

// Progress reports download progress for one file.
type Progress func(file string, done, total int64)

// Downloader fetches checkpoint bundles into a local cache.
type Downloader struct {
	// CacheDir defaults to $KEV_CACHE, then $XDG_CACHE_HOME/kev or ~/.cache/kev.
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
	if dir := os.Getenv(EnvCacheDir); dir != "" {
		return dir, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("kev: resolve cache dir: %w", err)
	}
	return filepath.Join(base, "kev"), nil
}

func (d *Downloader) cacheDir() (string, error) {
	if d.CacheDir != "" {
		return d.CacheDir, nil
	}
	return DefaultCacheDir()
}

func (d *Downloader) repo(ckpt Checkpoint) string {
	owner := d.RepoOwner
	if owner == "" {
		owner = DefaultRepoOwner
	}
	return owner + "/" + string(ckpt) + "-gguf"
}

func (d *Downloader) fileURL(ckpt Checkpoint, name string) string {
	endpoint := d.Endpoint
	if endpoint == "" {
		endpoint = os.Getenv(EnvHFEndpoint)
	}
	if endpoint == "" {
		endpoint = defaultHFEndpoint
	}
	revision := d.Revision
	if revision == "" {
		revision = defaultRevision
	}
	return fmt.Sprintf("%s/%s/resolve/%s/%s", strings.TrimRight(endpoint, "/"), d.repo(ckpt), revision, name)
}

func (d *Downloader) manifestDigest(ckpt Checkpoint) (string, bool) {
	if d.ManifestDigests != nil {
		if digest, ok := d.ManifestDigests[ckpt]; ok {
			return digest, true
		}
	}
	digest, ok := manifestDigests[ckpt]
	return digest, ok
}

func (d *Downloader) client() *http.Client {
	if d.HTTPClient != nil {
		return d.HTTPClient
	}
	return &http.Client{}
}

// Resolve returns a verified local bundle, downloading anything missing or
// corrupt. Local bundle directories (containing manifest.json) may be passed
// directly as the checkpoint to skip the network entirely.
func (d *Downloader) Resolve(ctx context.Context, ckpt Checkpoint) (*Bundle, error) {
	if info, err := os.Stat(string(ckpt)); err == nil && info.IsDir() {
		return loadLocalBundle(ckpt, string(ckpt))
	}
	pin, ok := d.manifestDigest(ckpt)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownCheckpoint, ckpt)
	}
	cache, err := d.cacheDir()
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(cache, string(ckpt))
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("kev: create cache dir: %w", err)
	}

	// The manifest is trusted only if it hashes to the compiled-in pin. A
	// cached manifest that matches needs no network; anything else is
	// fetched and checked before any file it names is touched.
	manifestPath := filepath.Join(dir, manifestName)
	pinned := ManifestFile{SHA256: pin}
	if verifyFile(manifestPath, pinned) != nil {
		if err := d.fetch(ctx, ckpt, manifestName, manifestPath, pinned); err != nil {
			return nil, err
		}
		if err := verifyFile(manifestPath, pinned); err != nil {
			_ = os.Remove(manifestPath)
			return nil, fmt.Errorf("%w: %s", ErrManifestUntrusted, ckpt)
		}
	}
	manifest, err := readManifest(manifestPath)
	if err != nil {
		return nil, err
	}
	for name, file := range manifest.Files {
		path := filepath.Join(dir, name)
		if verifyFile(path, file) == nil {
			continue
		}
		if err := d.fetch(ctx, ckpt, name, path, file); err != nil {
			return nil, err
		}
		if err := verifyFile(path, file); err != nil {
			return nil, err
		}
	}
	return bundleFrom(ckpt, dir, manifest)
}

// loadLocalBundle trusts a manifest on local disk (the caller pointed at
// it) but still verifies every file it lists.
func loadLocalBundle(ckpt Checkpoint, dir string) (*Bundle, error) {
	manifest, err := readManifest(filepath.Join(dir, manifestName))
	if err != nil {
		return nil, err
	}
	for name, file := range manifest.Files {
		if err := verifyFile(filepath.Join(dir, name), file); err != nil {
			return nil, err
		}
	}
	return bundleFrom(ckpt, dir, manifest)
}

func bundleFrom(ckpt Checkpoint, dir string, manifest Manifest) (*Bundle, error) {
	if manifest.Model == "" || manifest.Head == "" {
		return nil, errors.New("kev: manifest missing model or head entry")
	}
	return &Bundle{
		Checkpoint: ckpt,
		Dir:        dir,
		ModelPath:  filepath.Join(dir, manifest.Model),
		HeadPath:   filepath.Join(dir, manifest.Head),
		Manifest:   manifest,
	}, nil
}

func readManifest(path string) (Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("kev: read manifest: %w", err)
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, fmt.Errorf("kev: decode manifest: %w", err)
	}
	if manifest.Format != 1 {
		return Manifest{}, fmt.Errorf("kev: unsupported manifest format %d", manifest.Format)
	}
	return manifest, nil
}

// verifyFile checks size then sha256. An empty expected hash only checks
// existence, used for the manifest itself.
func verifyFile(path string, expected ManifestFile) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if expected.SHA256 == "" {
		return nil
	}
	if expected.Size != 0 && info.Size() != expected.Size {
		return fmt.Errorf("%w: %s size %d, want %d", ErrChecksum, filepath.Base(path), info.Size(), expected.Size)
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close() //nolint:errcheck
	hasher := sha256.New()
	if _, err := io.Copy(hasher, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(hasher.Sum(nil)); got != expected.SHA256 {
		return fmt.Errorf("%w: %s", ErrChecksum, filepath.Base(path))
	}
	return nil
}

// fetch downloads one file to path, resuming a partial download if present.
func (d *Downloader) fetch(ctx context.Context, ckpt Checkpoint, name, path string, expected ManifestFile) error {
	url := d.fileURL(ckpt, name)
	partial := path + partSuffix
	var offset int64
	if info, err := os.Stat(partial); err == nil {
		offset = info.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("kev: build request: %w", err)
	}
	if token := d.tokenValue(); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := d.client().Do(req)
	if err != nil {
		return fmt.Errorf("%w: %s: %w", ErrDownloadFailed, name, err)
	}
	defer resp.Body.Close() //nolint:errcheck

	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusOK:
		offset = 0
		flags |= os.O_TRUNC
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%w: %s: %s: %s", ErrDownloadFailed, name, resp.Status, strings.TrimSpace(string(body)))
	}
	total := expected.Size
	if total == 0 && resp.ContentLength > 0 {
		total = offset + resp.ContentLength
	}
	out, err := os.OpenFile(partial, flags, 0o600)
	if err != nil {
		return fmt.Errorf("kev: open %s: %w", partial, err)
	}
	writer := &progressWriter{w: out, done: offset, total: total, name: name, report: d.Progress}
	_, copyErr := io.Copy(writer, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		return fmt.Errorf("%w: %s: %w", ErrDownloadFailed, name, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("kev: close %s: %w", partial, closeErr)
	}
	if err := os.Rename(partial, path); err != nil {
		return fmt.Errorf("kev: finalize %s: %w", name, err)
	}
	return nil
}

func (d *Downloader) tokenValue() string {
	if d.Token != "" {
		return d.Token
	}
	return os.Getenv(EnvHFToken)
}

type progressWriter struct {
	w      io.Writer
	done   int64
	total  int64
	name   string
	report Progress
	last   time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.report != nil && (time.Since(p.last) > 200*time.Millisecond || p.done == p.total) {
		p.report(p.name, p.done, p.total)
		p.last = time.Now()
	}
	return n, err
}
