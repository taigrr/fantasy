package kev

import (
	"context"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/taigrr/fantasy/providers/internal/localgguf"
)

// LlamaCPPVersion is the llama.cpp build this package is tested against,
// pinned with the SHA-256 of its digest manifest so the manifest, every
// archive and every extracted file are verified before anything is loaded.
// It is shared with the other in-process providers; see localgguf.
const LlamaCPPVersion = localgguf.LlamaCPPVersion

const (
	// EnvLibPath overrides where the llama.cpp shared libraries live.
	EnvLibPath = localgguf.EnvLibPath
	// EnvVerbose leaves llama.cpp native logging enabled when set to 1.
	EnvVerbose = "KEV_VERBOSE"
	// EnvProcessor forces the llama.cpp backend flavour to install:
	// cpu, metal, cuda, rocm or vulkan.
	EnvProcessor = "KEV_PROCESSOR"

	seqID = llama.SeqId(0)
)

// Processor is a llama.cpp backend flavour.
type Processor = localgguf.Processor

// Supported backend flavours.
const (
	ProcessorAuto   = localgguf.ProcessorAuto
	ProcessorCPU    = localgguf.ProcessorCPU
	ProcessorMetal  = localgguf.ProcessorMetal
	ProcessorCUDA   = localgguf.ProcessorCUDA
	ProcessorROCm   = localgguf.ProcessorROCm
	ProcessorVulkan = localgguf.ProcessorVulkan
)

// Sentinel errors for library management. All wrap yzma's own where one
// exists so callers can also match on those.
var (
	// ErrLibrariesMissing means no llama.cpp install is present and automatic
	// installation was not enabled.
	ErrLibrariesMissing = localgguf.ErrLibrariesMissing
	// ErrLibrariesCorrupt means files on disk do not match the recorded
	// digests for the installed release.
	ErrLibrariesCorrupt = localgguf.ErrLibrariesCorrupt
	// ErrLibrariesOutdated means an install is present but for a different
	// llama.cpp release than LlamaCPPVersion.
	ErrLibrariesOutdated = localgguf.ErrLibrariesOutdated
	// ErrLibrariesUnavailable means no prebuilt llama.cpp exists for this
	// OS/arch/processor combination.
	ErrLibrariesUnavailable = localgguf.ErrLibrariesUnavailable
	// ErrDownloadVerification means a downloaded artifact did not match its
	// published digest. Nothing was installed.
	ErrDownloadVerification = localgguf.ErrDownloadVerification
)

// LibDir resolves the llama.cpp library directory: $YZMA_LIB, then
// $KEV_CACHE/lib/<tag> when KEV_CACHE is set, else the install shared by
// every in-process provider under the OS cache directory. Versioned
// directories let a new pinned release install beside an old one and let
// the old one be removed afterwards.
func LibDir() (string, error) {
	return localgguf.DefaultLibDir(EnvCacheDir)
}

// LibraryStatus describes what EnsureLibraries found or did.
type LibraryStatus = localgguf.LibraryStatus

// EnsureLibraries makes dir hold a verified install of LlamaCPPVersion for
// this machine, downloading only when needed. It never loads anything. See
// localgguf.EnsureLibraries for the exact sequence.
func EnsureLibraries(ctx context.Context, dir string, processor Processor, autoInstall bool) (LibraryStatus, error) {
	return localgguf.EnsureLibraries(ctx, dir, processor, EnvProcessor, autoInstall)
}

func detectProcessor() Processor {
	return localgguf.DetectProcessor(EnvProcessor)
}

// loadLibraries loads libllama once per process from dir. A second call
// with a different dir is an error: llama.cpp cannot be reloaded.
func loadLibraries(dir string) error {
	return localgguf.LoadLibraries(dir, EnvVerbose)
}
