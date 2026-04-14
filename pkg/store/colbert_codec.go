package store

import (
	"context"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// ColBERTCodec identifies how late-interaction segment vectors are stored.
type ColBERTCodec string

const (
	ColBERTCodecUnspecified ColBERTCodec = ""
	ColBERTCodecInt8        ColBERTCodec = "int8"
	ColBERTCodecPQ6         ColBERTCodec = "pq6"
)

func (c ColBERTCodec) String() string {
	if c == ColBERTCodecPQ6 {
		return string(ColBERTCodecPQ6)
	}
	return string(ColBERTCodecInt8)
}

// IsSpecified reports whether the codec came from an explicit configuration.
func (c ColBERTCodec) IsSpecified() bool {
	return c != ColBERTCodecUnspecified
}

// ParseColBERTCodec parses CLI/config input.
func ParseColBERTCodec(s string) ColBERTCodec {
	switch s {
	case "", "default":
		return ColBERTCodecUnspecified
	case "pq6", "pq":
		return ColBERTCodecPQ6
	default:
		return ColBERTCodecInt8
	}
}

// ResolveColBERTCodec chooses the effective codec, preferring explicit config,
// then persisted repo metadata, then the safe int8 default.
func ResolveColBERTCodec(requested, existing ColBERTCodec) ColBERTCodec {
	if requested.IsSpecified() {
		if requested == ColBERTCodecPQ6 {
			return ColBERTCodecPQ6
		}
		return ColBERTCodecInt8
	}
	if existing == ColBERTCodecPQ6 {
		return ColBERTCodecPQ6
	}
	return ColBERTCodecInt8
}

// ColBERTMetadataProvider exposes optional codec metadata to query-time code.
type ColBERTMetadataProvider interface {
	ColBERTCodec() ColBERTCodec
	ProductQuantizer() *util.ProductQuantizer
}

// ColBERTMetadataStore persists optional codec metadata during indexing.
type ColBERTMetadataStore interface {
	SaveColBERTMetadata(ctx context.Context, codec ColBERTCodec, pq *util.ProductQuantizer) error
	LoadColBERTMetadata(ctx context.Context) (ColBERTCodec, *util.ProductQuantizer, error)
}
