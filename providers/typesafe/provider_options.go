package typesafe

import (
	"encoding/json"

	"github.com/taigrr/fantasy"
)

// TypeProviderMetadata is the global type identifier for TypeSafe metadata.
const TypeProviderMetadata = Name + ".metadata"

func init() {
	fantasy.RegisterProviderType(TypeProviderMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderMetadata is response metadata from a TypeSafe-compatible server.
type ProviderMetadata struct {
	// LatencyMS is the server-reported latency, when present (Kev reports it).
	LatencyMS float64 `json:"latency_ms,omitempty"`
}

// Options implements fantasy.ProviderOptionsData.
func (*ProviderMetadata) Options() {}

// MarshalJSON implements json.Marshaler.
func (m ProviderMetadata) MarshalJSON() ([]byte, error) {
	type plain ProviderMetadata
	return fantasy.MarshalProviderType(TypeProviderMetadata, plain(m))
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *ProviderMetadata) UnmarshalJSON(data []byte) error {
	type plain ProviderMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ProviderMetadata(p)
	return nil
}
