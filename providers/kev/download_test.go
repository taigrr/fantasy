package kev

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPinnedCheckpointsAreComplete(t *testing.T) {
	for _, ckpt := range Checkpoints() {
		pin, ok := manifestDigests[ckpt]
		require.True(t, ok, "%s has no pinned manifest digest", ckpt)
		require.Len(t, pin, 64)
	}
}

func TestDownloaderRejectsUnpinnedCheckpoint(t *testing.T) {
	d := &Downloader{CacheDir: t.TempDir()}
	_, err := d.Resolve(context.Background(), Checkpoint("kev-nope"))
	require.ErrorIs(t, err, ErrUnknownCheckpoint)

	d.ManifestDigests = map[Checkpoint]string{"kev-nope": "00"}
	require.Equal(t, "taigrr/kev-nope-gguf", func() string { s, _ := d.shared("kev-nope"); return s.Repo("kev-nope") }())
}
