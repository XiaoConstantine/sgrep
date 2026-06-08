package store

import "testing"

func TestParseColBERTCodec_TQMSEAliases(t *testing.T) {
	for _, input := range []string{"tqmse", "tq-mse", "tq_mse", "tq-mse-4b"} {
		if got := ParseColBERTCodec(input); got != ColBERTCodecTQMSE {
			t.Fatalf("ParseColBERTCodec(%q) = %q, want %q", input, got, ColBERTCodecTQMSE)
		}
	}
}

func TestResolveColBERTCodec_DefaultsToTQMSE(t *testing.T) {
	if got := ResolveColBERTCodec(ColBERTCodecUnspecified, ColBERTCodecUnspecified); got != ColBERTCodecTQMSE {
		t.Fatalf("default codec = %q, want %q", got, ColBERTCodecTQMSE)
	}
	if got := ResolveColBERTCodec(ColBERTCodecInt8, ColBERTCodecTQMSE); got != ColBERTCodecInt8 {
		t.Fatalf("explicit int8 codec = %q, want %q", got, ColBERTCodecInt8)
	}
	if got := ResolveColBERTCodec(ColBERTCodecUnspecified, ColBERTCodecInt8); got != ColBERTCodecInt8 {
		t.Fatalf("existing int8 codec = %q, want %q", got, ColBERTCodecInt8)
	}
}
