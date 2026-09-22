package kev

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// LlamaCPPVersion is the llama.cpp build this package is tested against,
// pinned with the SHA-256 of its digest manifest so the manifest, every
// archive and every extracted file are verified before anything is loaded.
//
// It matches the release yzma v1.27.0 pins. Bump both together and re-run
// the parity tests.
const LlamaCPPVersion = "v0.4.1@sha256:e5fd75ea7d0f8f882de6f49968fb54fe19c4208882ae63b4d7eba97e9b747577"

const (
	// EnvLibPath overrides where the llama.cpp shared libraries live.
	EnvLibPath = "YZMA_LIB"
	// EnvVerbose leaves llama.cpp native logging enabled when set to 1.
	EnvVerbose = "KEV_VERBOSE"
	// EnvProcessor forces the llama.cpp backend flavour to install:
	// cpu, metal, cuda, rocm or vulkan.
	EnvProcessor = "KEV_PROCESSOR"

	seqID = llama.SeqId(0)
)

// Processor is a llama.cpp backend flavour.
type Processor string

// Supported backend flavours.
const (
	ProcessorAuto   Processor = ""
	ProcessorCPU    Processor = "cpu"
	ProcessorMetal  Processor = "metal"
	ProcessorCUDA   Processor = "cuda"
	ProcessorROCm   Processor = "rocm"
	ProcessorVulkan Processor = "vulkan"
)

// Sentinel errors for library management. All wrap yzma's own where one
// exists so callers can also match on those.
var (
	// ErrLibrariesMissing means no llama.cpp install is present and automatic
	// installation was not enabled.
	ErrLibrariesMissing = errors.New("kev: llama.cpp libraries are not installed")
	// ErrLibrariesCorrupt means files on disk do not match the recorded
	// digests for the installed release.
	ErrLibrariesCorrupt = errors.New("kev: llama.cpp libraries failed verification")
	// ErrLibrariesOutdated means an install is present but for a different
	// llama.cpp release than LlamaCPPVersion.
	ErrLibrariesOutdated = errors.New("kev: installed llama.cpp release does not match the pinned version")
	// ErrLibrariesUnavailable means no prebuilt llama.cpp exists for this
	// OS/arch/processor combination.
	ErrLibrariesUnavailable = errors.New("kev: no prebuilt llama.cpp for this platform")
	// ErrDownloadVerification means a downloaded artifact did not match its
	// published digest. Nothing was installed.
	ErrDownloadVerification = errors.New("kev: downloaded llama.cpp artifact failed digest verification")
)

var (
	libOnce sync.Once
	libErr  error
	libDir  string
)

// LibDir resolves the llama.cpp library directory: $YZMA_LIB, else
// <cache>/lib/<tag>. Versioned directories let a new pinned release install
// beside an old one and let the old one be removed afterwards.
func LibDir() (string, error) {
	if dir := os.Getenv(EnvLibPath); dir != "" {
		return dir, nil
	}
	cache, err := DefaultCacheDir()
	if err != nil {
		return "", err
	}
	tag, _, err := download.ParsePinnedVersion(LlamaCPPVersion)
	if err != nil {
		return "", fmt.Errorf("kev: bad pinned version: %w", err)
	}
	return filepath.Join(cache, "lib", tag), nil
}

// LibraryStatus describes what EnsureLibraries found or did.
type LibraryStatus struct {
	Dir       string
	Tag       string
	Processor Processor
	// Installed is true when this call downloaded and installed the libraries.
	Installed bool
	// Verified is true when on-disk files were hashed and matched the
	// release manifest.
	Verified bool
}

// EnsureLibraries makes dir hold a verified install of LlamaCPPVersion for
// this machine, downloading only when needed. It never loads anything.
//
// Sequence:
//  1. If an install record exists, hash every file against the release
//     manifest kept beside it. A match for the pinned tag returns without
//     touching the network.
//  2. A record for a different tag, or a failed hash, is treated as stale:
//     the directory is replaced in place by a fresh verified install.
//  3. With no record: if autoInstall is false return ErrLibrariesMissing;
//     otherwise download. The manifest must hash to the pinned digest and
//     every archive must match the manifest before any file is written.
func EnsureLibraries(ctx context.Context, dir string, processor Processor, autoInstall bool) (LibraryStatus, error) {
	tag, _, err := download.ParsePinnedVersion(LlamaCPPVersion)
	if err != nil {
		return LibraryStatus{}, fmt.Errorf("kev: bad pinned version: %w", err)
	}
	if processor == ProcessorAuto {
		processor = detectProcessor()
	}
	status := LibraryStatus{Dir: dir, Tag: tag, Processor: processor}

	// VerifyInstall hashes the files on disk against the pinned release's
	// manifest and does not trust the install record, so bytes from any
	// other release, or any tampering, show up as changed/missing files.
	report, verr := download.VerifyInstall(ctx, dir, LlamaCPPVersion)
	switch {
	case verr == nil && report.OK():
		status.Verified = true
		return status, nil
	case verr == nil:
		if !autoInstall {
			// Use the record only to phrase the error: a record naming
			// another tag most likely means an outdated install rather
			// than damage.
			if record, rerr := download.ReadInstallRecord(dir); rerr == nil && record.Tag != tag {
				return status, fmt.Errorf("%w: installed %s, want %s", ErrLibrariesOutdated, record.Tag, tag)
			}
			return status, fmt.Errorf("%w: %s", ErrLibrariesCorrupt, summarizeReport(report))
		}
	case errors.Is(verr, download.ErrNoInstallRecord):
		if !autoInstall {
			return status, fmt.Errorf("%w in %s", ErrLibrariesMissing, dir)
		}
	default:
		if !autoInstall {
			return status, fmt.Errorf("%w: %v", ErrLibrariesCorrupt, verr)
		}
	}

	if err := installLibraries(ctx, dir, processor); err != nil {
		return status, err
	}
	report, err = download.VerifyInstall(ctx, dir, LlamaCPPVersion)
	if err != nil {
		return status, fmt.Errorf("%w: post-install check: %v", ErrLibrariesCorrupt, err)
	}
	if !report.OK() {
		return status, fmt.Errorf("%w: post-install check: %s", ErrLibrariesCorrupt, summarizeReport(report))
	}
	status.Installed = true
	status.Verified = true
	return status, nil
}

func installLibraries(ctx context.Context, dir string, processor Processor) error {
	// Install into a staging directory, then swap, so a failure or a
	// concurrent reader never sees a half-written install.
	parent := filepath.Dir(dir)
	if err := os.MkdirAll(parent, 0o750); err != nil {
		return fmt.Errorf("kev: create lib parent: %w", err)
	}
	staging, err := os.MkdirTemp(parent, filepath.Base(dir)+".install-*")
	if err != nil {
		return fmt.Errorf("kev: create staging dir: %w", err)
	}
	defer os.RemoveAll(staging) //nolint:errcheck

	arch, err := download.ParseArch(runtime.GOARCH)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrLibrariesUnavailable, err)
	}
	goos, err := download.ParseOS(runtime.GOOS)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrLibrariesUnavailable, err)
	}
	proc, err := download.ParseProcessor(string(processor))
	if err != nil {
		return fmt.Errorf("%w: %v", ErrLibrariesUnavailable, err)
	}
	target := download.Target{Arch: arch, OS: goos, Processor: proc, Version: LlamaCPPVersion}
	err = download.Install(ctx, target, staging, nil, nil, download.WithVerify(download.VerifyRequired))
	switch {
	case err == nil:
	case errors.Is(err, download.ErrDigestMismatch), errors.Is(err, download.ErrDigestMissing), errors.Is(err, download.ErrInvalidDigest):
		return fmt.Errorf("%w: %v", ErrDownloadVerification, err)
	case errors.Is(err, download.ErrUnknownProcessor), errors.Is(err, download.ErrUnknownArch), errors.Is(err, download.ErrUnknownOS), errors.Is(err, download.ErrFileNotFound):
		return fmt.Errorf("%w (%s/%s/%s): %v", ErrLibrariesUnavailable, runtime.GOOS, runtime.GOARCH, processor, err)
	default:
		return fmt.Errorf("kev: install llama.cpp: %w", err)
	}

	// Replace the old install atomically where the filesystem allows it.
	old := dir + ".old"
	_ = os.RemoveAll(old)
	if _, statErr := os.Stat(dir); statErr == nil {
		if err := os.Rename(dir, old); err != nil {
			return fmt.Errorf("kev: retire old libraries: %w", err)
		}
	}
	if err := os.Rename(staging, dir); err != nil {
		_ = os.Rename(old, dir)
		return fmt.Errorf("kev: activate libraries: %w", err)
	}
	_ = os.RemoveAll(old)
	return nil
}

func summarizeReport(report *download.VerifyReport) string {
	var bad []string
	for _, file := range report.Files {
		if file.State != download.FileVerified {
			bad = append(bad, file.Name+"="+file.State.String())
		}
	}
	if len(bad) > 6 {
		bad = append(bad[:6], fmt.Sprintf("+%d more", len(bad)-6))
	}
	return strings.Join(bad, ", ")
}

// detectProcessor picks the best prebuilt backend for this machine. It only
// consults cheap, local signals; a wrong guess is recoverable with
// $KEV_PROCESSOR or WithProcessor.
func detectProcessor() Processor {
	if v := os.Getenv(EnvProcessor); v != "" {
		return Processor(strings.ToLower(v))
	}
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return ProcessorMetal
		}
		return ProcessorCPU
	case "linux", "windows":
		if ok, _ := download.HasCUDA(); ok {
			return ProcessorCUDA
		}
		if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" {
			if ok, _ := download.HasROCm(); ok {
				return ProcessorROCm
			}
		}
		if hasVulkan() {
			return ProcessorVulkan
		}
	}
	return ProcessorCPU
}

// hasVulkan reports whether a Vulkan loader appears to be present.
func hasVulkan() bool {
	switch runtime.GOOS {
	case "linux":
		for _, name := range []string{"libvulkan.so.1", "libvulkan.so"} {
			for _, dir := range []string{"/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/local/lib"} {
				if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
					return true
				}
			}
		}
		if _, err := exec.LookPath("vulkaninfo"); err == nil {
			return true
		}
		if entries, err := os.ReadDir("/usr/share/vulkan/icd.d"); err == nil && len(entries) > 0 {
			return true
		}
	case "windows":
		sysRoot := os.Getenv("SystemRoot")
		if sysRoot == "" {
			sysRoot = `C:\Windows`
		}
		if _, err := os.Stat(filepath.Join(sysRoot, "System32", "vulkan-1.dll")); err == nil { //nolint:gosec // read-only existence probe of a system path
			return true
		}
	}
	return false
}

// loadLibraries loads libllama once per process from dir. A second call
// with a different dir is an error: llama.cpp cannot be reloaded.
func loadLibraries(dir string) error {
	libOnce.Do(func() {
		libDir = dir
		if err := llama.Load(dir); err != nil {
			libErr = fmt.Errorf("kev: load llama.cpp from %q: %w", dir, err)
			return
		}
		if os.Getenv(EnvVerbose) != "1" {
			llama.LogSet(llama.LogSilent())
		}
		llama.Init()
	})
	if libErr != nil {
		return libErr
	}
	if libDir != dir {
		return fmt.Errorf("kev: llama.cpp already loaded from %q, cannot switch to %q in this process", libDir, dir)
	}
	return nil
}
