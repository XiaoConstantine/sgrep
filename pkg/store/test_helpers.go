//go:build !sqlite_vec
// +build !sqlite_vec

package store

// makeTestEmbedding creates a test embedding with consistent values.
func makeTestEmbedding(dims int, value float32) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = value
	}
	return vec
}

// itoa converts an int to a string (simple implementation for tests).
func itoa(n int) string {
	if n < 0 {
		return "-" + itoa(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}
