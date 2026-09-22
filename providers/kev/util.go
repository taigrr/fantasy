package kev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strconv"
)

// orderedObject is a JSON object with insertion order preserved, so state
// rendering matches Python's dict ordering.
type orderedObject struct {
	keys   []string
	values map[string]any
}

// decodeOrdered parses JSON into nested orderedObject / []any / string /
// float64 / bool / nil values.
func decodeOrdered(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	value, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("kev: trailing data after state JSON")
	}
	return value, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := &orderedObject{values: map[string]any{}}
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := keyTok.(string)
				if !ok {
					return nil, fmt.Errorf("kev: object key is %T", keyTok)
				}
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				if _, seen := obj.values[key]; !seen {
					obj.keys = append(obj.keys, key)
				}
				obj.values[key] = val
			}
			_, err := dec.Token() // '}'
			return obj, err
		case '[':
			var arr []any
			for dec.More() {
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, val)
			}
			_, err := dec.Token() // ']'
			if arr == nil {
				arr = []any{}
			}
			return arr, err
		}
		return nil, fmt.Errorf("kev: unexpected delimiter %v", t)
	case json.Number:
		return t, nil
	default:
		return t, nil
	}
}

// toOrdered marshals any Go value to JSON and decodes it order-preserving.
// Struct fields keep declaration order; Go maps are emitted sorted by
// encoding/json, which is the only deterministic choice available.
func toOrdered(v any) (any, error) {
	var data []byte
	switch raw := v.(type) {
	case json.RawMessage:
		data = raw
	case []byte:
		data = raw
	default:
		encoded, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("kev: marshal state: %w", err)
		}
		data = encoded
	}
	return decodeOrdered(data)
}

// formatNumber prints a JSON number as Python's str() would.
func formatNumber(n json.Number) string {
	if i, err := n.Int64(); err == nil {
		return strconv.FormatInt(i, 10)
	}
	f, err := n.Float64()
	if err != nil {
		return n.String()
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

func sortStrings(s []string) {
	slices.Sort(s)
}
