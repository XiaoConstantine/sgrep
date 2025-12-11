package util

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// ProductQuantizer implements Product Quantization for vector compression.
// PQ splits vectors into m subspaces and quantizes each to k centroids.
// This provides massive compression: 768-dim float32 (3072 bytes) -> 96 bytes (32x).
//
// For ColBERT reranking, we use:
//   - m=96 subspaces (768/96 = 8 dims per subspace)
//   - k=256 centroids (8-bit codes)
//   - Result: 96 bytes per vector
type ProductQuantizer struct {
	dims        int         // Original vector dimensions (e.g., 768)
	m           int         // Number of subspaces (e.g., 96)
	k           int         // Centroids per subspace (e.g., 256)
	subDims     int         // Dimensions per subspace (dims/m)
	centroids   [][]float32 // [m][k*subDims] - codebook for each subspace
	trained     bool
	trainingMu  sync.Mutex
}

// PQConfig holds Product Quantization configuration.
type PQConfig struct {
	Dims       int // Vector dimensions (default: 768)
	Subspaces  int // Number of subspaces m (default: 96)
	Centroids  int // Centroids per subspace k (default: 256)
	Iterations int // K-means iterations (default: 25)
}

// DefaultPQConfig returns optimal PQ config for 768-dim embeddings.
func DefaultPQConfig() PQConfig {
	return PQConfig{
		Dims:       768,
		Subspaces:  96,  // 768/96 = 8 dims per subspace
		Centroids:  256, // 8-bit codes
		Iterations: 25,
	}
}

// NewProductQuantizer creates a new PQ instance.
func NewProductQuantizer(cfg PQConfig) (*ProductQuantizer, error) {
	if cfg.Dims <= 0 {
		cfg.Dims = 768
	}
	if cfg.Subspaces <= 0 {
		cfg.Subspaces = 96
	}
	if cfg.Centroids <= 0 {
		cfg.Centroids = 256
	}
	if cfg.Dims%cfg.Subspaces != 0 {
		return nil, fmt.Errorf("dims (%d) must be divisible by subspaces (%d)", cfg.Dims, cfg.Subspaces)
	}
	if cfg.Centroids > 256 {
		return nil, fmt.Errorf("centroids (%d) must be <= 256 for 8-bit codes", cfg.Centroids)
	}

	subDims := cfg.Dims / cfg.Subspaces
	centroids := make([][]float32, cfg.Subspaces)
	for i := range centroids {
		centroids[i] = make([]float32, cfg.Centroids*subDims)
	}

	return &ProductQuantizer{
		dims:      cfg.Dims,
		m:         cfg.Subspaces,
		k:         cfg.Centroids,
		subDims:   subDims,
		centroids: centroids,
		trained:   false,
	}, nil
}

// Train builds the codebook from training vectors using k-means clustering.
// Requires at least k*10 training vectors for good results.
func (pq *ProductQuantizer) Train(vectors [][]float32, iterations int) error {
	pq.trainingMu.Lock()
	defer pq.trainingMu.Unlock()

	if len(vectors) < pq.k {
		return fmt.Errorf("need at least %d training vectors, got %d", pq.k, len(vectors))
	}
	if iterations <= 0 {
		iterations = 25
	}

	// Train each subspace independently
	for sub := 0; sub < pq.m; sub++ {
		// Extract subvectors for this subspace
		subvectors := make([][]float32, len(vectors))
		for i, vec := range vectors {
			if len(vec) != pq.dims {
				return fmt.Errorf("vector %d has wrong dims: %d (expected %d)", i, len(vec), pq.dims)
			}
			start := sub * pq.subDims
			subvectors[i] = vec[start : start+pq.subDims]
		}

		// Run k-means clustering
		centers := pq.kmeans(subvectors, pq.k, iterations)

		// Store centroids in flat array
		for c := 0; c < pq.k; c++ {
			copy(pq.centroids[sub][c*pq.subDims:(c+1)*pq.subDims], centers[c])
		}
	}

	pq.trained = true
	return nil
}

// kmeans runs k-means clustering on subvectors.
func (pq *ProductQuantizer) kmeans(subvectors [][]float32, k, iterations int) [][]float32 {
	n := len(subvectors)
	d := pq.subDims

	// Initialize centroids with k-means++ for better convergence
	centers := make([][]float32, k)
	centers[0] = make([]float32, d)
	copy(centers[0], subvectors[rand.Intn(n)])

	// K-means++ initialization
	distances := make([]float64, n)
	for c := 1; c < k; c++ {
		// Compute distance to nearest centroid for each point
		var totalDist float64
		for i, sv := range subvectors {
			minDist := math.MaxFloat64
			for j := 0; j < c; j++ {
				dist := pq.l2DistanceSquared(sv, centers[j])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist
			totalDist += minDist
		}

		// Sample proportional to distance squared
		r := rand.Float64() * totalDist
		var cumulative float64
		selected := 0
		for i, d := range distances {
			cumulative += d
			if cumulative >= r {
				selected = i
				break
			}
		}

		centers[c] = make([]float32, d)
		copy(centers[c], subvectors[selected])
	}

	// K-means iterations
	assignments := make([]int, n)
	counts := make([]int, k)
	newCenters := make([][]float32, k)
	for c := range newCenters {
		newCenters[c] = make([]float32, d)
	}

	for iter := 0; iter < iterations; iter++ {
		// Assign points to nearest centroid
		for i, sv := range subvectors {
			minDist := math.MaxFloat64
			minIdx := 0
			for c := 0; c < k; c++ {
				dist := pq.l2DistanceSquared(sv, centers[c])
				if dist < minDist {
					minDist = dist
					minIdx = c
				}
			}
			assignments[i] = minIdx
		}

		// Reset accumulators
		for c := 0; c < k; c++ {
			counts[c] = 0
			for j := 0; j < d; j++ {
				newCenters[c][j] = 0
			}
		}

		// Accumulate for new centroids
		for i, sv := range subvectors {
			c := assignments[i]
			counts[c]++
			for j := 0; j < d; j++ {
				newCenters[c][j] += sv[j]
			}
		}

		// Update centroids (handle empty clusters)
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				for j := 0; j < d; j++ {
					centers[c][j] = newCenters[c][j] / float32(counts[c])
				}
			}
			// Empty clusters keep previous centroid
		}
	}

	return centers
}

// l2DistanceSquared computes squared L2 distance between two vectors.
func (pq *ProductQuantizer) l2DistanceSquared(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return sum
}

// Encode quantizes a vector to PQ codes (m bytes).
func (pq *ProductQuantizer) Encode(vec []float32) ([]byte, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(vec) != pq.dims {
		return nil, fmt.Errorf("vector has wrong dims: %d (expected %d)", len(vec), pq.dims)
	}

	codes := make([]byte, pq.m)
	for sub := 0; sub < pq.m; sub++ {
		start := sub * pq.subDims
		subvec := vec[start : start+pq.subDims]

		// Find nearest centroid
		minDist := math.MaxFloat64
		minIdx := 0
		for c := 0; c < pq.k; c++ {
			centroid := pq.centroids[sub][c*pq.subDims : (c+1)*pq.subDims]
			dist := pq.l2DistanceSquared(subvec, centroid)
			if dist < minDist {
				minDist = dist
				minIdx = c
			}
		}
		codes[sub] = byte(minIdx)
	}

	return codes, nil
}

// EncodeBatch quantizes multiple vectors efficiently.
func (pq *ProductQuantizer) EncodeBatch(vectors [][]float32) ([][]byte, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	codes := make([][]byte, len(vectors))
	for i, vec := range vectors {
		c, err := pq.Encode(vec)
		if err != nil {
			return nil, fmt.Errorf("encode vector %d: %w", i, err)
		}
		codes[i] = c
	}
	return codes, nil
}

// Decode reconstructs an approximate vector from PQ codes.
func (pq *ProductQuantizer) Decode(codes []byte) ([]float32, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(codes) != pq.m {
		return nil, fmt.Errorf("codes has wrong length: %d (expected %d)", len(codes), pq.m)
	}

	vec := make([]float32, pq.dims)
	for sub := 0; sub < pq.m; sub++ {
		c := int(codes[sub])
		centroid := pq.centroids[sub][c*pq.subDims : (c+1)*pq.subDims]
		copy(vec[sub*pq.subDims:(sub+1)*pq.subDims], centroid)
	}

	return vec, nil
}

// DotProductWithCodes computes approximate dot product between query and PQ-encoded doc.
// Uses asymmetric distance computation (ADC) for better accuracy:
// query is kept in full precision, doc is reconstructed from PQ codes.
func (pq *ProductQuantizer) DotProductWithCodes(query []float32, codes []byte) (float64, error) {
	if !pq.trained {
		return 0, fmt.Errorf("quantizer not trained")
	}
	if len(query) != pq.dims {
		return 0, fmt.Errorf("query has wrong dims: %d (expected %d)", len(query), pq.dims)
	}
	if len(codes) != pq.m {
		return 0, fmt.Errorf("codes has wrong length: %d (expected %d)", len(codes), pq.m)
	}

	var dot float64
	for sub := 0; sub < pq.m; sub++ {
		c := int(codes[sub])
		centroid := pq.centroids[sub][c*pq.subDims : (c+1)*pq.subDims]
		querySubvec := query[sub*pq.subDims : (sub+1)*pq.subDims]

		for i := 0; i < pq.subDims; i++ {
			dot += float64(querySubvec[i]) * float64(centroid[i])
		}
	}

	return dot, nil
}

// PrecomputeQueryTable builds a lookup table for fast ADC scoring.
// Returns [m][k] table where table[sub][c] = dot(query_sub, centroid_sub_c).
// This is used when scoring many documents against the same query.
func (pq *ProductQuantizer) PrecomputeQueryTable(query []float32) ([][]float64, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(query) != pq.dims {
		return nil, fmt.Errorf("query has wrong dims: %d (expected %d)", len(query), pq.dims)
	}

	table := make([][]float64, pq.m)
	for sub := 0; sub < pq.m; sub++ {
		table[sub] = make([]float64, pq.k)
		querySubvec := query[sub*pq.subDims : (sub+1)*pq.subDims]

		for c := 0; c < pq.k; c++ {
			centroid := pq.centroids[sub][c*pq.subDims : (c+1)*pq.subDims]
			var dot float64
			for i := 0; i < pq.subDims; i++ {
				dot += float64(querySubvec[i]) * float64(centroid[i])
			}
			table[sub][c] = dot
		}
	}

	return table, nil
}

// DotProductWithTable computes dot product using precomputed lookup table.
// This is O(m) per document instead of O(dims) - a huge speedup for batch scoring.
func (pq *ProductQuantizer) DotProductWithTable(table [][]float64, codes []byte) float64 {
	var dot float64
	for sub := 0; sub < pq.m; sub++ {
		dot += table[sub][codes[sub]]
	}
	return dot
}

// SerializeCodebook serializes the trained codebook to bytes.
func (pq *ProductQuantizer) SerializeCodebook() ([]byte, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	// Header: dims(4) + m(4) + k(4) + subDims(4) = 16 bytes
	// Data: m * k * subDims * 4 bytes
	dataSize := pq.m * pq.k * pq.subDims * 4
	buf := make([]byte, 16+dataSize)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(pq.dims))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(pq.m))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(pq.k))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(pq.subDims))

	offset := 16
	for sub := 0; sub < pq.m; sub++ {
		for i := 0; i < pq.k*pq.subDims; i++ {
			binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(pq.centroids[sub][i]))
			offset += 4
		}
	}

	return buf, nil
}

// DeserializeCodebook loads a trained codebook from bytes.
func DeserializeCodebook(data []byte) (*ProductQuantizer, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("data too short for header")
	}

	dims := int(binary.LittleEndian.Uint32(data[0:4]))
	m := int(binary.LittleEndian.Uint32(data[4:8]))
	k := int(binary.LittleEndian.Uint32(data[8:12]))
	subDims := int(binary.LittleEndian.Uint32(data[12:16]))

	expectedSize := 16 + m*k*subDims*4
	if len(data) < expectedSize {
		return nil, fmt.Errorf("data too short: %d (expected %d)", len(data), expectedSize)
	}

	pq := &ProductQuantizer{
		dims:      dims,
		m:         m,
		k:         k,
		subDims:   subDims,
		centroids: make([][]float32, m),
		trained:   true,
	}

	offset := 16
	for sub := 0; sub < m; sub++ {
		pq.centroids[sub] = make([]float32, k*subDims)
		for i := 0; i < k*subDims; i++ {
			pq.centroids[sub][i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4
		}
	}

	return pq, nil
}

// CodeSize returns the size of PQ codes in bytes.
func (pq *ProductQuantizer) CodeSize() int {
	return pq.m
}

// IsTrained returns whether the quantizer has been trained.
func (pq *ProductQuantizer) IsTrained() bool {
	return pq.trained
}

// CompressionRatio returns the compression ratio vs float32.
func (pq *ProductQuantizer) CompressionRatio() float64 {
	return float64(pq.dims*4) / float64(pq.m)
}
