package util

import (
	"math"
	"math/rand"
	"testing"
)

func TestProductQuantizer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ test in short mode")
	}
	// Create random training vectors - use smaller set for CI speed
	rng := rand.New(rand.NewSource(42))
	nVectors := 300 // Reduced from 1000 for faster CI
	dims := 768
	vectors := make([][]float32, nVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dims)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()*2 - 1 // [-1, 1]
		}
		// Normalize
		vectors[i] = NormalizeVector(vectors[i])
	}

	// Create and train PQ
	cfg := DefaultPQConfig()
	pq, err := NewProductQuantizer(cfg)
	if err != nil {
		t.Fatalf("NewProductQuantizer: %v", err)
	}

	t.Logf("PQ config: dims=%d, subspaces=%d, centroids=%d", cfg.Dims, cfg.Subspaces, cfg.Centroids)
	t.Logf("Code size: %d bytes (vs %d for int8, %d for float32)", pq.CodeSize(), dims, dims*4)
	t.Logf("Compression ratio: %.1fx over float32", pq.CompressionRatio())

	err = pq.Train(vectors, 10) // Reduced iterations for faster CI
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	if !pq.IsTrained() {
		t.Fatal("Expected pq to be trained")
	}

	// Test encoding
	codes, err := pq.Encode(vectors[0])
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(codes) != cfg.Subspaces {
		t.Errorf("Expected %d codes, got %d", cfg.Subspaces, len(codes))
	}

	// Test decoding
	reconstructed, err := pq.Decode(codes)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(reconstructed) != dims {
		t.Errorf("Expected %d dims, got %d", dims, len(reconstructed))
	}

	// Check reconstruction quality
	original := vectors[0]
	var mse float64
	for i := range original {
		d := float64(original[i] - reconstructed[i])
		mse += d * d
	}
	mse /= float64(dims)
	t.Logf("Reconstruction MSE: %.6f", mse)
	t.Logf("Reconstruction RMSE: %.6f", math.Sqrt(mse))

	// Test dot product accuracy
	query := vectors[100]
	exactDot := DotProductUnrolled8(query, original)

	// ADC dot product
	adcDot, err := pq.DotProductWithCodes(query, codes)
	if err != nil {
		t.Fatalf("DotProductWithCodes: %v", err)
	}

	// Table-based dot product
	table, err := pq.PrecomputeQueryTable(query)
	if err != nil {
		t.Fatalf("PrecomputeQueryTable: %v", err)
	}
	tableDot := pq.DotProductWithTable(table, codes)

	t.Logf("Exact dot product: %.6f", exactDot)
	t.Logf("ADC dot product: %.6f (error: %.6f)", adcDot, math.Abs(exactDot-adcDot))
	t.Logf("Table dot product: %.6f (error: %.6f)", tableDot, math.Abs(exactDot-tableDot))

	// ADC and table should be identical
	if math.Abs(adcDot-tableDot) > 1e-9 {
		t.Errorf("ADC and table dot products differ: %.6f vs %.6f", adcDot, tableDot)
	}
}

func TestProductQuantizerSerialization(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ serialization test in short mode")
	}
	rng := rand.New(rand.NewSource(42))
	nVectors := 500
	dims := 768
	vectors := make([][]float32, nVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dims)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()*2 - 1
		}
	}

	// Create, train, and serialize
	pq, _ := NewProductQuantizer(DefaultPQConfig())
	if err := pq.Train(vectors, 10); err != nil {
		t.Fatalf("Train: %v", err)
	}

	data, err := pq.SerializeCodebook()
	if err != nil {
		t.Fatalf("SerializeCodebook: %v", err)
	}
	t.Logf("Codebook size: %d bytes (%.2f KB)", len(data), float64(len(data))/1024)

	// Deserialize
	pq2, err := DeserializeCodebook(data)
	if err != nil {
		t.Fatalf("DeserializeCodebook: %v", err)
	}

	// Verify they produce same codes
	codes1, _ := pq.Encode(vectors[0])
	codes2, _ := pq2.Encode(vectors[0])

	for i := range codes1 {
		if codes1[i] != codes2[i] {
			t.Errorf("Code mismatch at %d: %d vs %d", i, codes1[i], codes2[i])
		}
	}
}

func TestPQRankingAccuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ ranking accuracy test in short mode")
	}
	// Test if PQ preserves ranking order (most important for reranking)
	rng := rand.New(rand.NewSource(42))
	nDocs := 500 // Need at least 256 for k=256 centroids
	dims := 768
	docs := make([][]float32, nDocs)
	for i := range docs {
		docs[i] = make([]float32, dims)
		for j := range docs[i] {
			docs[i][j] = rng.Float32()*2 - 1
		}
		docs[i] = NormalizeVector(docs[i])
	}

	// Train PQ
	pq, err := NewProductQuantizer(DefaultPQConfig())
	if err != nil {
		t.Fatalf("NewProductQuantizer: %v", err)
	}
	if err := pq.Train(docs, 25); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Encode all docs
	docCodes := make([][]byte, nDocs)
	for i, doc := range docs {
		var err error
		docCodes[i], err = pq.Encode(doc)
		if err != nil {
			t.Fatalf("Encode doc %d: %v", i, err)
		}
	}

	// Random query
	query := make([]float32, dims)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}
	query = NormalizeVector(query)

	// Compute exact and PQ scores
	exactScores := make([]float64, nDocs)
	pqScores := make([]float64, nDocs)
	table, err := pq.PrecomputeQueryTable(query)
	if err != nil {
		t.Fatalf("PrecomputeQueryTable: %v", err)
	}

	for i := range docs {
		exactScores[i] = DotProductUnrolled8(query, docs[i])
		pqScores[i] = pq.DotProductWithTable(table, docCodes[i])
	}

	// Find top-10 by exact and PQ
	exactTop10 := topKIndices(exactScores, 10)
	pqTop10 := topKIndices(pqScores, 10)

	// Count how many of PQ top-10 are in exact top-10
	overlap := 0
	for _, pqIdx := range pqTop10 {
		for _, exactIdx := range exactTop10 {
			if pqIdx == exactIdx {
				overlap++
				break
			}
		}
	}

	t.Logf("Top-10 overlap: %d/10 (%.0f%% recall)", overlap, float64(overlap)*10)

	// Also check correlation
	var sumRankDiff float64
	for i := 0; i < nDocs; i++ {
		exactRank := findRank(exactScores, i)
		pqRank := findRank(pqScores, i)
		sumRankDiff += math.Abs(float64(exactRank - pqRank))
	}
	avgRankDiff := sumRankDiff / float64(nDocs)
	t.Logf("Average rank difference: %.2f", avgRankDiff)
}

func topKIndices(scores []float64, k int) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	// Simple selection sort for top-k
	for i := 0; i < k && i < len(scores); i++ {
		maxIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[indices[j]] > scores[indices[maxIdx]] {
				maxIdx = j
			}
		}
		indices[i], indices[maxIdx] = indices[maxIdx], indices[i]
	}
	if k > len(scores) {
		k = len(scores)
	}
	return indices[:k]
}

func findRank(scores []float64, idx int) int {
	targetScore := scores[idx]
	rank := 1
	for i, s := range scores {
		if i != idx && s > targetScore {
			rank++
		}
	}
	return rank
}

func BenchmarkPQEncode(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	nVectors := 1000
	dims := 768
	vectors := make([][]float32, nVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dims)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()*2 - 1
		}
	}

	pq, _ := NewProductQuantizer(DefaultPQConfig())
	_ = pq.Train(vectors[:500], 10)

	testVec := vectors[500]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = pq.Encode(testVec)
	}
}

func BenchmarkPQDotProduct(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	nVectors := 1000
	dims := 768
	vectors := make([][]float32, nVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dims)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()*2 - 1
		}
	}

	pq, _ := NewProductQuantizer(DefaultPQConfig())
	_ = pq.Train(vectors[:500], 10)

	query := vectors[500]
	docCodes, _ := pq.Encode(vectors[501])

	b.Run("ADC", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = pq.DotProductWithCodes(query, docCodes)
		}
	})

	table, _ := pq.PrecomputeQueryTable(query)
	b.Run("Table", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			pq.DotProductWithTable(table, docCodes)
		}
	})
}

func BenchmarkPQVsInt8(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dims := 768
	query := make([]float32, dims)
	doc := make([]float32, dims)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
		doc[i] = rng.Float32()*2 - 1
	}
	query = NormalizeVector(query)
	doc = NormalizeVector(doc)

	// Int8 quantized
	docInt8, scale, min := QuantizeInt8(doc)

	// PQ encoded
	vectors := make([][]float32, 500)
	for i := range vectors {
		vectors[i] = make([]float32, dims)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()*2 - 1
		}
	}
	pq, _ := NewProductQuantizer(DefaultPQConfig())
	_ = pq.Train(vectors, 10)
	docPQ, _ := pq.Encode(doc)
	table, _ := pq.PrecomputeQueryTable(query)

	b.Run("Float32_Dot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DotProductUnrolled8(query, doc)
		}
	})

	b.Run("Int8_Dot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DotProductInt8Unrolled8(query, docInt8, scale, min)
		}
	})

	b.Run("PQ_Table_Dot", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			pq.DotProductWithTable(table, docPQ)
		}
	})

	// Report sizes
	b.Logf("Float32: %d bytes", dims*4)
	b.Logf("Int8: %d bytes", dims)
	b.Logf("PQ: %d bytes", len(docPQ))
}
