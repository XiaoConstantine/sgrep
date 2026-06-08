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
	ColBERTCodecTQMSE       ColBERTCodec = "tqmse"
)

func (c ColBERTCodec) String() string {
	switch c {
	case ColBERTCodecPQ6:
		return string(ColBERTCodecPQ6)
	case ColBERTCodecTQMSE:
		return string(ColBERTCodecTQMSE)
	case ColBERTCodecUnspecified:
		return string(ColBERTCodecUnspecified)
	default:
		return string(ColBERTCodecInt8)
	}
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
	case "tqmse", "tq-mse", "tq_mse", "tq-mse-4b", "tqmse4":
		return ColBERTCodecTQMSE
	default:
		return ColBERTCodecInt8
	}
}

// ResolveColBERTCodec chooses the effective codec, preferring explicit config,
// then persisted repo metadata, then the compact TQ-MSE default.
func ResolveColBERTCodec(requested, existing ColBERTCodec) ColBERTCodec {
	if requested.IsSpecified() {
		switch requested {
		case ColBERTCodecPQ6:
			return ColBERTCodecPQ6
		case ColBERTCodecTQMSE:
			return ColBERTCodecTQMSE
		default:
			return ColBERTCodecInt8
		}
	}
	switch existing {
	case ColBERTCodecPQ6:
		return ColBERTCodecPQ6
	case ColBERTCodecTQMSE:
		return ColBERTCodecTQMSE
	case ColBERTCodecInt8:
		return ColBERTCodecInt8
	default:
		return ColBERTCodecTQMSE
	}
}

// ColBERTMetadataProvider exposes optional codec metadata to query-time code.
type ColBERTMetadataProvider interface {
	ColBERTCodec() ColBERTCodec
	ProductQuantizer() *util.ProductQuantizer
	TQMSEQuantizer() *util.TQMSEQuantizer
}

// ColBERTMetadataStore persists optional codec metadata during indexing.
type ColBERTMetadataStore interface {
	SaveColBERTMetadata(ctx context.Context, codec ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) error
	LoadColBERTMetadata(ctx context.Context) (ColBERTCodec, *util.ProductQuantizer, *util.TQMSEQuantizer, error)
}
