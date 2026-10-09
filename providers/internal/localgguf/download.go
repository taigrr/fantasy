package localgguf

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

const (
	// EnvHFEndpoint overrides the Hugging Face host (mirrors, offline caches).
	EnvHFEndpoint = "HF_ENDPOINT"
	// EnvHFToken supplies a token for private repos or higher rate limits.
	EnvHFToken = "HF_TOKEN"

	// DefaultRepoOwner is the Hugging Face account hosting the GGUF bundles.
	DefaultRepoOwner = "taigrr"
	// RepoSuffix is appended to a checkpoint name to form its repository.
	RepoSuffix = "-gguf"

	defaultHFEndpoint = "https://huggingface.co"
	// ManifestName is the bundle manifest file name.
	ManifestName    = "manifest.json"
	partSuffix      = ".part"
	defaultRevision = "main"
	manifestFormat  = 1
)

// Sentinel errors for checkpoint downloads.
var (
	// ErrChecksum means a bundle file does not match its manifest.
	ErrChecksum = errors.New("checksum mismatch")
	// ErrManifestUntrusted means a bundle manifest does not hash to the
	// digest compiled into the provider for that checkpoint. Nothing from
	// it is used.
	ErrManifestUntrusted = errors.New("checkpoint manifest does not match pinned digest")
	// ErrUnknownCheckpoint means the checkpoint is neither a published name
	// with a pinned manifest digest nor a local bundle directory.
	ErrUnknownCheckpoint = errors.New("unknown checkpoint")
	// ErrDownloadFailed wraps HTTP and I/O failures while fetching a bundle.
	ErrDownloadFailed = errors.New("checkpoint download failed")
)

// Manifest describes a bundle. It is written by the conversion scripts in
// gojev/tools; fields a given model family does not use are left empty.
type Manifest struct {
	Format int `json:"format"`
	// Decision names the decision head family ("kev", "clef"); empty means
	// kev for bundles written before the field existed.
	Decision      string                  `json:"decision,omitempty"`
	Run           string                  `json:"run"`
	Base          string                  `json:"base"`
	Temperature   float64                 `json:"temperature,omitempty"`
	HiddenSize    int                     `json:"hidden_size"`
	HeadDim       int                     `json:"head_dim,omitempty"`
	ContextWindow int                     `json:"context_window,omitempty"`
	Model         string                  `json:"model"`
	Head          string                  `json:"head,omitempty"`
	Files         map[string]ManifestFile `json:"files"`
}

// ManifestFile is one entry in Manifest.Files.
type ManifestFile struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// Bundle is a resolved, verified checkpoint on local disk.
type Bundle struct {
	Checkpoint string
	Dir        string
	ModelPath  string
	// HeadPath is empty when the manifest names no separate head file.
	HeadPath string
	Manifest Manifest
}

// Progress reports download progress for one file.
type Progress func(file string, done, total int64)

// Downloader fetches checkpoint bundles into a local cache.
type Downloader struct {
	// CacheDir is where bundles live, one subdirectory per checkpoint.
	CacheDir string
	// RepoOwner defaults to DefaultRepoOwner; the repo is <owner>/<checkpoint>-gguf.
	RepoOwner string
	// Revision defaults to "main".
	Revision string
	// Endpoint defaults to $HF_ENDPOINT or https://huggingface.co.
	Endpoint string
	// ManifestDigests pins the SHA-256 of manifest.json per checkpoint. A
	// manifest served from Hugging Face is only used if it hashes to the
	// value here, and every file it lists is then verified against it.
	ManifestDigests map[string]string
	// Token defaults to $HF_TOKEN.
	Token string
	// HTTPClient defaults to a client with no overall timeout (files are large).
	HTTPClient *http.Client
	// Progress is optional.
	Progress Progress
}

// DefaultCacheDir resolves a cache directory: $<envName> when set, else the
// OS user cache directory joined with subdir.
func DefaultCacheDir(envName, subdir string) (string, error) {
	if dir := os.Getenv(envName); dir != "" {
		return dir, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("resolve cache dir: %w", err)
	}
	return filepath.Join(base, subdir), nil
}

// Repo returns the Hugging Face repository for a checkpoint.
func (d *Downloader) Repo(ckpt string) string {
	owner := d.RepoOwner
	if owner == "" {
		owner = DefaultRepoOwner
	}
	return owner + "/" + ckpt + RepoSuffix
}

func (d *Downloader) fileURL(ckpt, name string) string {
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
	return fmt.Sprintf("%s/%s/resolve/%s/%s", strings.TrimRight(endpoint, "/"), d.Repo(ckpt), revision, name)
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
func (d *Downloader) Resolve(ctx context.Context, ckpt string) (*Bundle, error) {
	if info, err := os.Stat(ckpt); err == nil && info.IsDir() {
		return LoadLocalBundle(ckpt, ckpt)
	}
	pin, ok := d.ManifestDigests[ckpt]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownCheckpoint, ckpt)
	}
	if d.CacheDir == "" {
		return nil, errors.New("downloader has no cache directory")
	}
	dir := filepath.Join(d.CacheDir, ckpt)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("create cache dir: %w", err)
	}

	// The manifest is trusted only if it hashes to the compiled-in pin. A
	// cached manifest that matches needs no network; anything else is
	// fetched and checked before any file it names is touched.
	manifestPath := filepath.Join(dir, ManifestName)
	pinned := ManifestFile{SHA256: pin}
	if VerifyFile(manifestPath, pinned) != nil {
		if err := d.fetch(ctx, ckpt, ManifestName, manifestPath, pinned); err != nil {
			return nil, err
		}
		if err := VerifyFile(manifestPath, pinned); err != nil {
			_ = os.Remove(manifestPath)
			return nil, fmt.Errorf("%w: %s", ErrManifestUntrusted, ckpt)
		}
	}
	manifest, err := ReadManifest(manifestPath)
	if err != nil {
		return nil, err
	}
	for name, file := range manifest.Files {
		path := filepath.Join(dir, name)
		if VerifyFile(path, file) == nil {
			continue
		}
		if err := d.fetch(ctx, ckpt, name, path, file); err != nil {
			return nil, err
		}
		if err := VerifyFile(path, file); err != nil {
			return nil, err
		}
	}
	return bundleFrom(ckpt, dir, manifest)
}

// LoadLocalBundle trusts a manifest on local disk (the caller pointed at
// it) but still verifies every file it lists.
func LoadLocalBundle(ckpt, dir string) (*Bundle, error) {
	manifest, err := ReadManifest(filepath.Join(dir, ManifestName))
	if err != nil {
		return nil, err
	}
	for name, file := range manifest.Files {
		if err := VerifyFile(filepath.Join(dir, name), file); err != nil {
			return nil, err
		}
	}
	return bundleFrom(ckpt, dir, manifest)
}

func bundleFrom(ckpt, dir string, manifest Manifest) (*Bundle, error) {
	if manifest.Model == "" {
		return nil, errors.New("manifest missing model entry")
	}
	bundle := &Bundle{
		Checkpoint: ckpt,
		Dir:        dir,
		ModelPath:  filepath.Join(dir, manifest.Model),
		Manifest:   manifest,
	}
	if manifest.Head != "" {
		bundle.HeadPath = filepath.Join(dir, manifest.Head)
	}
	return bundle, nil
}

// ReadManifest parses and checks a manifest file.
func ReadManifest(path string) (Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("read manifest: %w", err)
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w", err)
	}
	if manifest.Format != manifestFormat {
		return Manifest{}, fmt.Errorf("unsupported manifest format %d", manifest.Format)
	}
	return manifest, nil
}

// VerifyFile checks size then sha256. An empty expected hash only checks
// existence, used for the manifest itself.
func VerifyFile(path string, expected ManifestFile) error {
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
func (d *Downloader) fetch(ctx context.Context, ckpt, name, path string, expected ManifestFile) error {
	url := d.fileURL(ckpt, name)
	partial := path + partSuffix
	var offset int64
	if info, err := os.Stat(partial); err == nil {
		offset = info.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
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
		return fmt.Errorf("open %s: %w", partial, err)
	}
	writer := &progressWriter{w: out, done: offset, total: total, name: name, report: d.Progress}
	_, copyErr := io.Copy(writer, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		return fmt.Errorf("%w: %s: %w", ErrDownloadFailed, name, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %s: %w", partial, closeErr)
	}
	if err := os.Rename(partial, path); err != nil {
		return fmt.Errorf("finalize %s: %w", name, err)
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
