package kev

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/stretchr/testify/require"
)

// These tests exercise EnsureLibraries against a real, verified install and
// deliberately damaged copies of it. They need the pinned release present
// (run the integration test once) and are skipped otherwise.
func installedLibDir(t *testing.T) string {
	t.Helper()
	dir, err := LibDir()
	require.NoError(t, err)
	if _, err := download.ReadInstallRecord(dir); err != nil {
		t.Skipf("no verified llama.cpp install at %s; run the integration test first", dir)
	}
	return dir
}

func copyDir(t *testing.T, src, dst string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(dst, 0o750))
	entries, err := os.ReadDir(src)
	require.NoError(t, err)
	for _, entry := range entries {
		from := filepath.Join(src, entry.Name())
		to := filepath.Join(dst, entry.Name())
		if entry.IsDir() {
			copyDir(t, from, to)
			continue
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(from)
			require.NoError(t, err)
			require.NoError(t, os.Symlink(target, to))
			continue
		}
		data, err := os.ReadFile(from)
		require.NoError(t, err)
		info, err := entry.Info()
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(to, data, info.Mode()))
	}
}

func TestEnsureLibraries_VerifiedInstallIsOfflineFastPath(t *testing.T) {
	dir := installedLibDir(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := time.Now()
	status, err := EnsureLibraries(ctx, dir, ProcessorAuto, false)
	require.NoError(t, err)
	require.True(t, status.Verified)
	require.False(t, status.Installed)
	require.Less(t, time.Since(start), 5*time.Second, "verification must not hit the network")
}

func TestEnsureLibraries_MissingWithoutAutoInstall(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "libs")
	_, err := EnsureLibraries(context.Background(), dir, ProcessorCPU, false)
	require.ErrorIs(t, err, ErrLibrariesMissing)
}

func TestEnsureLibraries_DetectsTamperedLibrary(t *testing.T) {
	src := installedLibDir(t)
	dir := filepath.Join(t.TempDir(), "libs")
	copyDir(t, src, dir)

	// Flip one byte in the main library.
	lib := filepath.Join(dir, download.LibraryName("darwin"))
	if _, err := os.Stat(lib); err != nil {
		lib = filepath.Join(dir, download.LibraryName("linux"))
	}
	data, err := os.ReadFile(lib)
	require.NoError(t, err)
	data[len(data)/2] ^= 0xff
	require.NoError(t, os.WriteFile(lib, data, 0o600))

	_, err = EnsureLibraries(context.Background(), dir, ProcessorAuto, false)
	require.ErrorIs(t, err, ErrLibrariesCorrupt)
	require.Contains(t, err.Error(), "changed")
}

func TestEnsureLibraries_DetectsDeletedFile(t *testing.T) {
	src := installedLibDir(t)
	dir := filepath.Join(t.TempDir(), "libs")
	copyDir(t, src, dir)
	record, err := download.ReadInstallRecord(dir)
	require.NoError(t, err)
	require.NotEmpty(t, record.Assets)

	// Remove a file the manifest says should be there.
	report, err := download.VerifyInstall(context.Background(), dir, LlamaCPPVersion)
	require.NoError(t, err)
	require.NotEmpty(t, report.Files)
	require.NoError(t, os.Remove(filepath.Join(dir, report.Files[len(report.Files)-1].Name)))

	_, err = EnsureLibraries(context.Background(), dir, ProcessorAuto, false)
	require.ErrorIs(t, err, ErrLibrariesCorrupt)
	require.Contains(t, err.Error(), "missing")
}

func TestEnsureLibraries_ForgedManifestCannotLaunderTamperedLibrary(t *testing.T) {
	src := installedLibDir(t)
	dir := filepath.Join(t.TempDir(), "libs")
	copyDir(t, src, dir)

	// An attacker who can write the lib dir can rewrite the library AND the
	// cached manifest to match. The pinned manifest digest compiled into this
	// package must reject the forged manifest, and the genuine one must then
	// reject the library.
	lib := filepath.Join(dir, download.LibraryName("darwin"))
	if _, err := os.Stat(lib); err != nil {
		lib = filepath.Join(dir, download.LibraryName("linux"))
	}
	data, err := os.ReadFile(lib)
	require.NoError(t, err)
	data[len(data)/2] ^= 0xff
	require.NoError(t, os.WriteFile(lib, data, 0o600))

	forged := sha256Hex(data)
	manifestPath := filepath.Join(dir, download.InstallManifestName)
	manifest, err := os.ReadFile(manifestPath)
	require.NoError(t, err)
	// Swap every occurrence of the real library digest with the forged one.
	realDigest, err := hashFileHex(filepath.Join(src, filepath.Base(lib)))
	require.NoError(t, err)
	require.Contains(t, string(manifest), realDigest, "manifest should record the library digest")
	manifest = []byte(strings.ReplaceAll(string(manifest), realDigest, forged))
	require.NoError(t, os.WriteFile(manifestPath, manifest, 0o600))

	_, err = EnsureLibraries(context.Background(), dir, ProcessorAuto, false)
	require.ErrorIs(t, err, ErrLibrariesCorrupt, "forged manifest must not launder a tampered library")
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hashFileHex(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return sha256Hex(data), nil
}

func TestEnsureLibraries_ReplacesStaleTagInPlace(t *testing.T) {
	if os.Getenv("KEV_NETWORK_TESTS") != "1" {
		t.Skip("set KEV_NETWORK_TESTS=1 to exercise the download path")
	}
	src := installedLibDir(t)
	dir := filepath.Join(t.TempDir(), "libs")
	copyDir(t, src, dir)

	// Simulate an install from an older release: the record names another
	// tag and the library bytes differ from the pinned release.
	record, err := download.ReadInstallRecord(dir)
	require.NoError(t, err)
	record.Tag = "v0.0.1"
	require.NoError(t, download.WriteInstallRecord(dir, *record))
	lib := filepath.Join(dir, download.LibraryName("darwin"))
	if _, err := os.Stat(lib); err != nil {
		lib = filepath.Join(dir, download.LibraryName("linux"))
	}
	require.NoError(t, os.WriteFile(lib, []byte("old release"), 0o600))

	_, err = EnsureLibraries(context.Background(), dir, ProcessorAuto, false)
	require.ErrorIs(t, err, ErrLibrariesOutdated)

	status, err := EnsureLibraries(context.Background(), dir, ProcessorAuto, true)
	require.NoError(t, err)
	require.True(t, status.Installed)
	require.True(t, status.Verified)
	fresh, err := download.ReadInstallRecord(dir)
	require.NoError(t, err)
	require.Equal(t, status.Tag, fresh.Tag)
	_, err = os.Stat(dir + ".old")
	require.True(t, os.IsNotExist(err), "retired install must be cleaned up")
}

func TestDetectProcessorHonoursEnv(t *testing.T) {
	t.Setenv(EnvProcessor, "VULKAN")
	require.Equal(t, ProcessorVulkan, detectProcessor())
	t.Setenv(EnvProcessor, "")
	require.NotEqual(t, ProcessorAuto, detectProcessor())
}
