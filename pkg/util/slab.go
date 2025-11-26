package util

import (
	"math"
)

// VectorSlab is a pre-allocated buffer for vector operations.
// Avoids GC pressure during hot search paths (inspired by fzf's slab pattern).
type VectorSlab struct {
	distances []float64 // Pre-allocated distance buffer
	indices   []int     // Pre-allocated index buffer
	scratch   []float32 // Temp buffer for vector ops
}

// NewVectorSlab creates a slab with pre-allocated buffers.
func NewVectorSlab(maxDocs int, dims int) *VectorSlab {
	return &VectorSlab{
		distances: make([]float64, maxDocs),
		indices:   make([]int, maxDocs),
		scratch:   make([]float32, dims),
	}
}

// Reset prepares the slab for reuse.
func (s *VectorSlab) Reset(n int) {
	if n > len(s.distances) {
		s.distances = make([]float64, n)
		s.indices = make([]int, n)
	}
}

// Distances returns the distance buffer (up to n elements).
func (s *VectorSlab) Distances(n int) []float64 {
	s.Reset(n)
	return s.distances[:n]
}

// Indices returns the index buffer (up to n elements).
func (s *VectorSlab) Indices(n int) []int {
	s.Reset(n)
	return s.indices[:n]
}

// SlabPool manages a pool of VectorSlabs for parallel operations.
type SlabPool struct {
	slabs   []*VectorSlab
	maxDocs int
	dims    int
}

// NewSlabPool creates a pool with numPartitions slabs.
func NewSlabPool(numPartitions, maxDocsPerPartition, dims int) *SlabPool {
	slabs := make([]*VectorSlab, numPartitions)
	for i := range slabs {
		slabs[i] = NewVectorSlab(maxDocsPerPartition, dims)
	}
	return &SlabPool{
		slabs:   slabs,
		maxDocs: maxDocsPerPartition,
		dims:    dims,
	}
}

// Get returns a slab for the given partition index.
func (p *SlabPool) Get(partition int) *VectorSlab {
	if partition < len(p.slabs) {
		return p.slabs[partition]
	}
	// Fallback: create new slab if partition exceeds pool size
	return NewVectorSlab(p.maxDocs, p.dims)
}

// L2DistanceSlab computes L2 distance using slab's scratch buffer.
func L2DistanceSlab(a, b []float32, scratch []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// L2DistanceBatch computes L2 distances for multiple vectors using pre-allocated buffer.
// Results are written to the distances slice (must be len(vectors)).
func L2DistanceBatch(query []float32, vectors [][]float32, distances []float64) {
	for i, vec := range vectors {
		if len(query) != len(vec) {
			distances[i] = math.MaxFloat64
			continue
		}
		var sum float64
		for j := range query {
			d := float64(query[j] - vec[j])
			sum += d * d
		}
		distances[i] = math.Sqrt(sum)
	}
}

// TopKIndices returns indices of k smallest values in distances (partial sort).
// Uses indices buffer for zero allocation.
func TopKIndices(distances []float64, indices []int, k int) []int {
	n := len(distances)
	if k > n {
		k = n
	}

	// Initialize indices
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	// Partial selection sort for top-k (O(n*k) but no allocation)
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < n; j++ {
			if distances[indices[j]] < distances[indices[minIdx]] {
				minIdx = j
			}
		}
		indices[i], indices[minIdx] = indices[minIdx], indices[i]
	}

	return indices[:k]
}
