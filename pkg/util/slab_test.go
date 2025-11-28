package util

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestNewVectorSlab(t *testing.T) {
	slab := NewVectorSlab(1000, 768)

	if len(slab.distances) != 1000 {
		t.Errorf("expected distances length 1000, got %d", len(slab.distances))
	}
	if len(slab.indices) != 1000 {
		t.Errorf("expected indices length 1000, got %d", len(slab.indices))
	}
	if len(slab.scratch) != 768 {
		t.Errorf("expected scratch length 768, got %d", len(slab.scratch))
	}
}

func TestVectorSlab_Reset(t *testing.T) {
	slab := NewVectorSlab(100, 768)

	slab.Reset(50)
	if len(slab.distances) != 100 {
		t.Error("Reset should not shrink buffers")
	}

	slab.Reset(200)
	if len(slab.distances) < 200 {
		t.Error("Reset should grow buffers when n exceeds capacity")
	}
}

func TestVectorSlab_Distances(t *testing.T) {
	slab := NewVectorSlab(100, 768)

	d50 := slab.Distances(50)
	if len(d50) != 50 {
		t.Errorf("expected length 50, got %d", len(d50))
	}

	d200 := slab.Distances(200)
	if len(d200) != 200 {
		t.Errorf("expected length 200, got %d", len(d200))
	}
}

func TestVectorSlab_Indices(t *testing.T) {
	slab := NewVectorSlab(100, 768)

	i50 := slab.Indices(50)
	if len(i50) != 50 {
		t.Errorf("expected length 50, got %d", len(i50))
	}

	i200 := slab.Indices(200)
	if len(i200) != 200 {
		t.Errorf("expected length 200, got %d", len(i200))
	}
}

func TestNewSlabPool(t *testing.T) {
	pool := NewSlabPool(4, 1000, 768)

	if len(pool.slabs) != 4 {
		t.Errorf("expected 4 slabs, got %d", len(pool.slabs))
	}
	if pool.maxDocs != 1000 {
		t.Errorf("expected maxDocs 1000, got %d", pool.maxDocs)
	}
	if pool.dims != 768 {
		t.Errorf("expected dims 768, got %d", pool.dims)
	}
}

func TestSlabPool_Get(t *testing.T) {
	pool := NewSlabPool(4, 1000, 768)

	slab0 := pool.Get(0)
	slab3 := pool.Get(3)

	if slab0 == nil || slab3 == nil {
		t.Error("Get should return non-nil slabs for valid partitions")
	}

	slabOver := pool.Get(10)
	if slabOver == nil {
		t.Error("Get should create fallback slab for out-of-range partition")
	}
}

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 0.0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 1.0,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: 2.0,
		},
		{
			name:     "similar vectors",
			a:        []float32{1, 1, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0 - 1.0/1.4142135623730951, // 1 - 1/sqrt(2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineDistance(tt.a, tt.b)
			if math.Abs(got-tt.expected) > 1e-6 {
				t.Errorf("CosineDistance() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestCosineDistanceBatch(t *testing.T) {
	query := []float32{1, 0, 0}
	vectors := [][]float32{
		{1, 0, 0},  // identical
		{0, 1, 0},  // orthogonal
		{-1, 0, 0}, // opposite
	}
	distances := make([]float64, 3)

	CosineDistanceBatch(query, vectors, distances)

	expected := []float64{0.0, 1.0, 2.0}
	for i, exp := range expected {
		if math.Abs(distances[i]-exp) > 1e-6 {
			t.Errorf("CosineDistanceBatch[%d] = %v, want %v", i, distances[i], exp)
		}
	}
}

func TestL2DistanceSlab(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1.0, 2.0, 3.0},
			b:        []float32{1.0, 2.0, 3.0},
			expected: 0.0,
		},
		{
			name:     "different vectors",
			a:        []float32{0.0, 0.0, 0.0},
			b:        []float32{1.0, 0.0, 0.0},
			expected: 1.0,
		},
		{
			name:     "3-4-5 triangle",
			a:        []float32{0.0, 0.0},
			b:        []float32{3.0, 4.0},
			expected: 5.0,
		},
		{
			name:     "length mismatch",
			a:        []float32{1.0, 2.0},
			b:        []float32{1.0, 2.0, 3.0},
			expected: math.MaxFloat64,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scratch := make([]float32, max(len(tt.a), len(tt.b)))
			got := L2DistanceSlab(tt.a, tt.b, scratch)
			if math.Abs(got-tt.expected) > 1e-6 {
				t.Errorf("L2DistanceSlab() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestL2DistanceBatch(t *testing.T) {
	query := []float32{0.0, 0.0, 0.0}
	vectors := [][]float32{
		{0.0, 0.0, 0.0}, // distance = 0
		{1.0, 0.0, 0.0}, // distance = 1
		{0.0, 3.0, 4.0}, // distance = 5
	}
	distances := make([]float64, len(vectors))

	L2DistanceBatch(query, vectors, distances)

	expected := []float64{0.0, 1.0, 5.0}
	for i, exp := range expected {
		if math.Abs(distances[i]-exp) > 1e-6 {
			t.Errorf("distances[%d] = %v, want %v", i, distances[i], exp)
		}
	}
}

func TestL2DistanceBatch_DimensionMismatch(t *testing.T) {
	query := []float32{0.0, 0.0, 0.0}
	vectors := [][]float32{
		{0.0, 0.0},       // wrong dims
		{1.0, 0.0, 0.0},  // correct dims
		{0.0, 3.0, 4.0, 5.0}, // wrong dims
	}
	distances := make([]float64, len(vectors))

	L2DistanceBatch(query, vectors, distances)

	if distances[0] != math.MaxFloat64 {
		t.Errorf("expected MaxFloat64 for dimension mismatch at index 0")
	}
	if math.Abs(distances[1]-1.0) > 1e-6 {
		t.Errorf("expected 1.0 for correct match at index 1")
	}
	if distances[2] != math.MaxFloat64 {
		t.Errorf("expected MaxFloat64 for dimension mismatch at index 2")
	}
}

func TestTopKIndices(t *testing.T) {
	tests := []struct {
		name      string
		distances []float64
		k         int
		expected  []int
	}{
		{
			name:      "simple top-3",
			distances: []float64{5.0, 1.0, 3.0, 2.0, 4.0},
			k:         3,
			expected:  []int{1, 3, 2}, // indices sorted by distance: 1.0, 2.0, 3.0
		},
		{
			name:      "k larger than n",
			distances: []float64{3.0, 1.0},
			k:         5,
			expected:  []int{1, 0},
		},
		{
			name:      "k equals n",
			distances: []float64{3.0, 1.0, 2.0},
			k:         3,
			expected:  []int{1, 2, 0},
		},
		{
			name:      "single element",
			distances: []float64{5.0},
			k:         1,
			expected:  []int{0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := make([]int, len(tt.distances))
			got := TopKIndices(tt.distances, indices, tt.k)

			if len(got) != len(tt.expected) {
				t.Errorf("TopKIndices() returned %d indices, want %d", len(got), len(tt.expected))
				return
			}

			for i, exp := range tt.expected {
				if got[i] != exp {
					t.Errorf("TopKIndices()[%d] = %d, want %d", i, got[i], exp)
				}
			}
		})
	}
}

func TestSlabPool_Concurrent(t *testing.T) {
	pool := NewSlabPool(8, 1000, 768)
	var wg sync.WaitGroup

	for partition := 0; partition < 8; partition++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			slab := pool.Get(p)
			distances := slab.Distances(100)
			for j := range distances {
				distances[j] = float64(j)
			}
		}(partition)
	}

	wg.Wait()
}

// Benchmarks

func BenchmarkL2DistanceSlab_768dims(b *testing.B) {
	a := make([]float32, 768)
	b2 := make([]float32, 768)
	scratch := make([]float32, 768)

	for i := range a {
		a[i] = rand.Float32()
		b2[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2DistanceSlab(a, b2, scratch)
	}
}

func BenchmarkL2DistanceBatch_1000vectors(b *testing.B) {
	query := make([]float32, 768)
	vectors := make([][]float32, 1000)
	distances := make([]float64, 1000)

	for i := range query {
		query[i] = rand.Float32()
	}
	for i := range vectors {
		vectors[i] = make([]float32, 768)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2DistanceBatch(query, vectors, distances)
	}
}

func BenchmarkTopKIndices_1000vectors_Top10(b *testing.B) {
	distances := make([]float64, 1000)
	indices := make([]int, 1000)

	for i := range distances {
		distances[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TopKIndices(distances, indices, 10)
	}
}

func BenchmarkSlabPool_Get(b *testing.B) {
	pool := NewSlabPool(16, 10000, 768)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.Get(i % 16)
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
