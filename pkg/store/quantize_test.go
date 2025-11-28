package store

import (
	"math"
	"testing"
)

func TestQuantizeToInt8Unit(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []int8
	}{
		{
			name:     "zeros",
			input:    []float32{0, 0, 0},
			expected: []int8{0, 0, 0},
		},
		{
			name:     "ones",
			input:    []float32{1, 1, 1},
			expected: []int8{127, 127, 127},
		},
		{
			name:     "negative ones",
			input:    []float32{-1, -1, -1},
			expected: []int8{-127, -127, -127},
		},
		{
			name:     "mixed",
			input:    []float32{-1.0, -0.5, 0, 0.5, 1.0},
			expected: []int8{-127, -64, 0, 64, 127},
		},
		{
			name:     "clamped above",
			input:    []float32{2.0},
			expected: []int8{127},
		},
		{
			name:     "clamped below",
			input:    []float32{-2.0},
			expected: []int8{-127},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QuantizeToInt8Unit(tt.input)
			if len(got) != len(tt.expected) {
				t.Errorf("length mismatch: got %d, want %d", len(got), len(tt.expected))
				return
			}
			for i := range got {
				// Allow for rounding differences
				if abs(int(got[i])-int(tt.expected[i])) > 1 {
					t.Errorf("index %d: got %d, want %d", i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func TestQuantizeToBinary(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []byte
	}{
		{
			name:     "all positive",
			input:    []float32{1, 1, 1, 1, 1, 1, 1, 1},
			expected: []byte{0xFF},
		},
		{
			name:     "all negative",
			input:    []float32{-1, -1, -1, -1, -1, -1, -1, -1},
			expected: []byte{0x00},
		},
		{
			name:     "alternating",
			input:    []float32{1, -1, 1, -1, 1, -1, 1, -1},
			expected: []byte{0xAA}, // 10101010
		},
		{
			name:     "16 elements",
			input:    []float32{1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1},
			expected: []byte{0xFF, 0x00},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QuantizeToBinary(tt.input)
			if len(got) != len(tt.expected) {
				t.Errorf("length mismatch: got %d, want %d", len(got), len(tt.expected))
				return
			}
			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("byte %d: got %02x, want %02x", i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func TestSerializeDeserializeInt8(t *testing.T) {
	original := []int8{-128, -64, 0, 64, 127}
	blob := SerializeInt8(original)
	recovered := DeserializeInt8(blob)

	if len(recovered) != len(original) {
		t.Fatalf("length mismatch: got %d, want %d", len(recovered), len(original))
	}

	for i := range original {
		if recovered[i] != original[i] {
			t.Errorf("index %d: got %d, want %d", i, recovered[i], original[i])
		}
	}
}

func TestQuantizeRoundTrip(t *testing.T) {
	// Test that quantization + dequantization preserves approximate values
	original := []float32{-1.0, -0.5, 0, 0.5, 1.0}
	int8Vec := QuantizeToInt8Unit(original)
	recovered := DequantizeInt8(int8Vec)

	for i := range original {
		diff := math.Abs(float64(original[i] - recovered[i]))
		if diff > 0.02 { // Allow 2% error
			t.Errorf("index %d: original %f, recovered %f, diff %f", i, original[i], recovered[i], diff)
		}
	}
}

func TestParseQuantizationMode(t *testing.T) {
	tests := []struct {
		input    string
		expected QuantizationMode
	}{
		{"none", QuantizeNone},
		{"int8", QuantizeInt8},
		{"scalar", QuantizeInt8},
		{"binary", QuantizeBinary},
		{"bit", QuantizeBinary},
		{"invalid", QuantizeNone},
		{"", QuantizeNone},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := ParseQuantizationMode(tt.input)
			if got != tt.expected {
				t.Errorf("got %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestQuantizationModeString(t *testing.T) {
	tests := []struct {
		mode     QuantizationMode
		expected string
	}{
		{QuantizeNone, "none"},
		{QuantizeInt8, "int8"},
		{QuantizeBinary, "binary"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			got := tt.mode.String()
			if got != tt.expected {
				t.Errorf("got %s, want %s", got, tt.expected)
			}
		})
	}
}
