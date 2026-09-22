package evalhttp

import (
	"fmt"
	"slices"
	"strconv"
)

// LegendToLevels converts a wire legend keyed "0", "1", ... into an ordered
// slice.
func LegendToLevels(legend map[string]string) ([]string, error) {
	return orderedByIndex(legend)
}

// LevelProbabilitiesFromMap converts a wire map keyed "0", "1", ... into an
// ordered slice.
func LevelProbabilitiesFromMap(probabilities map[string]float64) ([]float64, error) {
	return orderedByIndex(probabilities)
}

func orderedByIndex[T any](m map[string]T) ([]T, error) {
	if len(m) == 0 {
		return nil, nil
	}
	keys := make([]int, 0, len(m))
	for k := range m {
		idx, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("non-integer level key %q", k)
		}
		keys = append(keys, idx)
	}
	slices.Sort(keys)
	out := make([]T, len(keys))
	for i, idx := range keys {
		if idx != i {
			return nil, fmt.Errorf("non-contiguous level keys: expected %d, got %d", i, idx)
		}
		out[i] = m[strconv.Itoa(idx)]
	}
	return out, nil
}
