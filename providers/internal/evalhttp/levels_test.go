package evalhttp

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLegendToLevels(t *testing.T) {
	t.Parallel()
	levels, err := LegendToLevels(map[string]string{"2": "c", "0": "a", "1": "b"})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c"}, levels)

	_, err = LegendToLevels(map[string]string{"0": "a", "2": "c"})
	require.Error(t, err)
	_, err = LegendToLevels(map[string]string{"x": "a"})
	require.Error(t, err)

	levels, err = LegendToLevels(nil)
	require.NoError(t, err)
	require.Nil(t, levels)
}
