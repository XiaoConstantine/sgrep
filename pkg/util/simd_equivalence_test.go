//go:build goexperiment.simd && !arm64

package util

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

func TestSIMDBuildDotProductMatchesScalar(t *testing.T) {
	const dims = 768

	huge := make([]float32, 32)
	for i := range huge {
		huge[i] = 1e20
	}
	assertDotMatchesScalar(t, "large finite products", huge, huge)

	uniform := make([]float32, dims)
	value := float32(1 / math.Sqrt(dims))
	for i := range uniform {
		uniform[i] = value
	}
	assertDotMatchesScalar(t, "uniform normalized vector", uniform, uniform)

	rng := rand.New(rand.NewSource(47))
	query := randomNormalizedVector(rng, dims)
	for i := range 512 {
		doc := randomNormalizedVector(rng, dims)
		assertDotMatchesScalar(t, "random normalized document "+strconv.Itoa(i), query, doc)
	}

	tailQuery := randomNormalizedVector(rng, dims+1)
	tailDoc := randomNormalizedVector(rng, dims+1)
	assertDotMatchesScalar(t, "vector tail", tailQuery, tailDoc)
}

func assertDotMatchesScalar(t *testing.T, name string, a, b []float32) {
	t.Helper()
	got := DotProductUnrolled8(a, b)
	want := dotProductUnrolled8Scalar(a, b)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s: DotProductUnrolled8 = %.12g, scalar = %.12g", name, got, want)
	}
}

func randomNormalizedVector(rng *rand.Rand, dims int) []float32 {
	values := make([]float32, dims)
	for i := range values {
		values[i] = rng.Float32()*2 - 1
	}
	return NormalizeVector(values)
}
