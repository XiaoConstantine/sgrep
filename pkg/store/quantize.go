package store

import (
	"math"
)

// QuantizationMode specifies how embeddings are stored.
type QuantizationMode int

const (
	// QuantizeNone stores embeddings as float32 (4 bytes per element).
	QuantizeNone QuantizationMode = iota
	// QuantizeInt8 stores embeddings as int8 (1 byte per element, 4x smaller).
	QuantizeInt8
	// QuantizeBinary stores embeddings as bits (1 bit per element, 32x smaller).
	QuantizeBinary
)

func (m QuantizationMode) String() string {
	switch m {
	case QuantizeInt8:
		return "int8"
	case QuantizeBinary:
		return "binary"
	default:
		return "none"
	}
}

// ParseQuantizationMode parses a string to QuantizationMode.
func ParseQuantizationMode(s string) QuantizationMode {
	switch s {
	case "int8", "scalar":
		return QuantizeInt8
	case "binary", "bit":
		return QuantizeBinary
	default:
		return QuantizeNone
	}
}

// QuantizeToInt8 converts float32 embeddings to int8 using min-max scaling.
// This maps the range [min, max] to [-128, 127].
func QuantizeToInt8(vec []float32) []int8 {
	if len(vec) == 0 {
		return nil
	}

	// Find min and max
	minVal, maxVal := vec[0], vec[0]
	for _, v := range vec[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Handle constant vectors
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		rangeVal = 1
	}

	result := make([]int8, len(vec))
	scale := 255.0 / float64(rangeVal)
	for i, v := range vec {
		// Scale to [0, 255] then shift to [-128, 127]
		scaled := (float64(v-minVal) * scale) - 128
		result[i] = int8(math.Round(scaled))
	}
	return result
}

// QuantizeToInt8Unit converts normalized float32 embeddings [-1, 1] to int8.
// This is more efficient for already-normalized embeddings.
func QuantizeToInt8Unit(vec []float32) []int8 {
	result := make([]int8, len(vec))
	for i, v := range vec {
		// Clamp to [-1, 1] and scale to [-128, 127]
		clamped := v
		if clamped < -1 {
			clamped = -1
		} else if clamped > 1 {
			clamped = 1
		}
		result[i] = int8(clamped * 127)
	}
	return result
}

// QuantizeToBinary converts float32 embeddings to binary (1 bit per element).
// Positive values become 1, negative values become 0.
func QuantizeToBinary(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}

	// Calculate number of bytes needed (ceiling division by 8)
	nBytes := (len(vec) + 7) / 8
	result := make([]byte, nBytes)

	for i, v := range vec {
		if v > 0 {
			byteIdx := i / 8
			bitIdx := uint(7 - (i % 8)) // MSB first
			result[byteIdx] |= 1 << bitIdx
		}
	}
	return result
}

// SerializeInt8 serializes int8 vector to bytes for sqlite-vec.
func SerializeInt8(vec []int8) []byte {
	result := make([]byte, len(vec))
	for i, v := range vec {
		result[i] = byte(v)
	}
	return result
}

// DeserializeInt8 deserializes bytes back to int8 vector.
func DeserializeInt8(blob []byte) []int8 {
	result := make([]int8, len(blob))
	for i, b := range blob {
		result[i] = int8(b)
	}
	return result
}

// DequantizeInt8 converts int8 back to float32 for query vectors.
// Uses unit scaling (assumes original was normalized to [-1, 1]).
func DequantizeInt8(vec []int8) []float32 {
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = float32(v) / 127.0
	}
	return result
}
