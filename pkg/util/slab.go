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

// NormalizeVector normalizes a vector to unit length (L2 norm = 1).
// This enables fast cosine distance via dot product: distance = 1 - dot(a, b).
// Modifies the vector in place and returns it.
func NormalizeVector(v []float32) []float32 {
	var norm float64
	for _, val := range v {
		norm += float64(val) * float64(val)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v
	}

	invNorm := float32(1.0 / norm)
	for i := range v {
		v[i] *= invNorm
	}
	return v
}

// NormalizeVectorCopy returns a normalized copy of the vector.
func NormalizeVectorCopy(v []float32) []float32 {
	result := make([]float32, len(v))
	copy(result, v)
	return NormalizeVector(result)
}

// CosineDistance computes cosine distance (1 - cosine_similarity).
// Returns 0 for identical vectors, 2 for opposite vectors.
// For pre-normalized vectors, use DotProductDistance instead (faster).
func CosineDistance(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.MaxFloat64
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return math.MaxFloat64
	}

	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - similarity
}

// DotProductDistance computes cosine distance for PRE-NORMALIZED vectors.
// distance = 1 - dot(a, b), where both vectors have unit length.
// This is ~4x faster than CosineDistance since no norm computation is needed.
func DotProductDistance(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return 1.0 - dot
}

// DotProductUnrolled8 computes dot product with 8-way loop unrolling.
// Exploits CPU pipelining by breaking data dependencies between iterations.
// For 768-dim vectors (common embedding size): 768/8 = 96 iterations.
// Returns the raw dot product (not distance). For similarity of normalized
// vectors, this equals cosine similarity directly.
func DotProductUnrolled8(a, b []float32) float64 {
	n := len(a)
	if n != len(b) {
		return 0
	}

	var s0, s1, s2, s3, s4, s5, s6, s7 float64

	// Unrolled main loop (8 elements per iteration)
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += float64(a[i]) * float64(b[i])
		s1 += float64(a[i+1]) * float64(b[i+1])
		s2 += float64(a[i+2]) * float64(b[i+2])
		s3 += float64(a[i+3]) * float64(b[i+3])
		s4 += float64(a[i+4]) * float64(b[i+4])
		s5 += float64(a[i+5]) * float64(b[i+5])
		s6 += float64(a[i+6]) * float64(b[i+6])
		s7 += float64(a[i+7]) * float64(b[i+7])
	}

	// Handle remainder (for dimensions not divisible by 8)
	for ; i < n; i++ {
		s0 += float64(a[i]) * float64(b[i])
	}

	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

// CosineDistanceBatch computes cosine distances for multiple vectors.
// Results are written to the distances slice (must be len(vectors)).
// For pre-normalized vectors, use DotProductDistanceBatch instead (faster).
func CosineDistanceBatch(query []float32, vectors [][]float32, distances []float64) {
	// Pre-compute query norm
	var queryNorm float64
	for _, v := range query {
		queryNorm += float64(v) * float64(v)
	}
	queryNorm = math.Sqrt(queryNorm)

	if queryNorm == 0 {
		for i := range distances {
			distances[i] = math.MaxFloat64
		}
		return
	}

	for i, vec := range vectors {
		if len(query) != len(vec) {
			distances[i] = math.MaxFloat64
			continue
		}

		var dotProduct, vecNorm float64
		for j := range query {
			dotProduct += float64(query[j]) * float64(vec[j])
			vecNorm += float64(vec[j]) * float64(vec[j])
		}

		if vecNorm == 0 {
			distances[i] = math.MaxFloat64
			continue
		}

		similarity := dotProduct / (queryNorm * math.Sqrt(vecNorm))
		distances[i] = 1.0 - similarity
	}
}

// DotProductDistanceBatch computes cosine distances for PRE-NORMALIZED vectors.
// distance = 1 - dot(query, vec) for each vector.
// This is ~4x faster than CosineDistanceBatch since no norm computation is needed.
// IMPORTANT: Both query and all vectors must be pre-normalized to unit length.
func DotProductDistanceBatch(query []float32, vectors [][]float32, distances []float64) {
	for i, vec := range vectors {
		if len(query) != len(vec) {
			distances[i] = math.MaxFloat64
			continue
		}

		var dot float64
		for j := range query {
			dot += float64(query[j]) * float64(vec[j])
		}
		distances[i] = 1.0 - dot
	}
}

// QuantizeInt8 quantizes a float32 vector to int8 using min-max scaling.
// Returns the quantized values along with scale and offset for dequantization.
// Formula: quantized = round((value - min) / scale) - 128
// This maps the range [min, max] to [-128, 127].
func QuantizeInt8(v []float32) ([]int8, float32, float32) {
	if len(v) == 0 {
		return nil, 0, 0
	}

	// Find min and max
	min, max := v[0], v[0]
	for _, val := range v[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	// Handle edge case where all values are the same
	scale := max - min
	if scale == 0 {
		scale = 1.0
	}
	scale = scale / 255.0 // Map to 256 levels

	result := make([]int8, len(v))
	for i, val := range v {
		// Normalize to [0, 255], then shift to [-128, 127]
		normalized := (val - min) / scale
		result[i] = int8(normalized - 128)
	}

	return result, scale, min
}

// DequantizeInt8 converts int8 quantized values back to float32.
// Formula: value = (quantized + 128) * scale + min
func DequantizeInt8(q []int8, scale, min float32) []float32 {
	result := make([]float32, len(q))
	for i, val := range q {
		result[i] = (float32(val)+128)*scale + min
	}
	return result
}

// DequantizeInt8Into converts int8 quantized values into a pre-allocated float32 buffer.
// More efficient than DequantizeInt8 when processing many vectors.
func DequantizeInt8Into(q []int8, scale, min float32, out []float32) {
	for i, val := range q {
		out[i] = (float32(val)+128)*scale + min
	}
}

// DotProductInt8 computes approximate dot product between a float32 query vector
// and a quantized int8 document vector. Dequantizes on-the-fly for efficiency.
// For ColBERT, we need high precision for the query (float32) but can use
// quantized storage for documents.
func DotProductInt8(query []float32, doc []int8, scale, min float32) float64 {
	n := len(query)
	if n != len(doc) {
		return 0
	}

	var sum float64
	for i := 0; i < n; i++ {
		// Dequantize on the fly: docVal = (int8 + 128) * scale + min
		docVal := (float32(doc[i])+128)*scale + min
		sum += float64(query[i]) * float64(docVal)
	}
	return sum
}

// DotProductInt8Unrolled8 computes dot product with 8-way unrolling for int8 doc vectors.
// Combines dequantization with dot product for better cache efficiency.
func DotProductInt8Unrolled8(query []float32, doc []int8, scale, min float32) float64 {
	n := len(query)
	if n != len(doc) {
		return 0
	}

	var s0, s1, s2, s3, s4, s5, s6, s7 float64

	// Unrolled main loop
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += float64(query[i]) * float64((float32(doc[i])+128)*scale+min)
		s1 += float64(query[i+1]) * float64((float32(doc[i+1])+128)*scale+min)
		s2 += float64(query[i+2]) * float64((float32(doc[i+2])+128)*scale+min)
		s3 += float64(query[i+3]) * float64((float32(doc[i+3])+128)*scale+min)
		s4 += float64(query[i+4]) * float64((float32(doc[i+4])+128)*scale+min)
		s5 += float64(query[i+5]) * float64((float32(doc[i+5])+128)*scale+min)
		s6 += float64(query[i+6]) * float64((float32(doc[i+6])+128)*scale+min)
		s7 += float64(query[i+7]) * float64((float32(doc[i+7])+128)*scale+min)
	}

	// Handle remainder
	for ; i < n; i++ {
		s0 += float64(query[i]) * float64((float32(doc[i])+128)*scale+min)
	}

	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
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
