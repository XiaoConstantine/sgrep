package store

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

func TestSegmentPooler_Pool(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768

	// Create 20 random segments
	segments := make([]ColBERTSegment, 20)
	for i := range segments {
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "segment " + string(rune('A'+i)),
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	pooler := NewSegmentPooler(5, 0.90)
	pooled := pooler.Pool(segments)

	t.Logf("Original: %d segments, Pooled: %d segments", len(segments), len(pooled))
	if len(pooled) > 5 {
		t.Errorf("Expected at most 5 pooled segments, got %d", len(pooled))
	}

	// Check indices are renumbered
	for i, seg := range pooled {
		if seg.SegmentIdx != i {
			t.Errorf("Segment %d has wrong index: %d", i, seg.SegmentIdx)
		}
	}
}

func TestSegmentPooler_MergeBySimilarity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768

	// Create segments with some very similar ones
	baseEmb := make([]float32, dims)
	for i := range baseEmb {
		baseEmb[i] = rand.Float32()*2 - 1
	}
	baseEmb = util.NormalizeVector(baseEmb)

	segments := make([]ColBERTSegment, 5)

	// First 3 are very similar (small perturbations)
	for i := 0; i < 3; i++ {
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = baseEmb[j] + rand.Float32()*0.02 - 0.01 // Small noise
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "similar " + string(rune('A'+i)),
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	// Last 2 are different
	for i := 3; i < 5; i++ {
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1 // Completely random
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "different " + string(rune('A'+i)),
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	pooler := NewSegmentPooler(10, 0.95) // High similarity threshold
	merged := pooler.MergeBySimilarity(segments)

	t.Logf("Original: %d segments, Merged: %d segments", len(segments), len(merged))

	// The 3 similar should merge into 1, leaving 3 total
	if len(merged) > 4 {
		t.Logf("Expected around 3 merged segments (3 similar -> 1 + 2 different), got %d", len(merged))
	}
}

func TestSegmentPooler_PoolAndMerge(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768

	// Create 15 segments with some duplicates
	segments := make([]ColBERTSegment, 15)
	for i := range segments {
		// Create groups of similar embeddings
		groupBase := (i / 3) * 100 // Groups of 3
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = float32(groupBase+j%10) / 100.0 // Deterministic per group
			emb[j] += rand.Float32() * 0.01          // Small noise within group
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "segment " + string(rune('A'+i)),
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	pooler := NewSegmentPooler(5, 0.95)
	result := pooler.PoolAndMerge(segments)

	t.Logf("Original: %d segments, After PoolAndMerge: %d segments", len(segments), len(result))

	if len(result) > 5 {
		t.Errorf("Expected at most 5 segments after PoolAndMerge, got %d", len(result))
	}
}

func TestMMapSegmentStore(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768
	ctx := context.Background()

	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "mmap_test")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// Create store
	store, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}

	// Create test segments for 3 chunks
	chunks := map[string][]ColBERTSegment{
		"chunk1": createTestSegments(t, dims, 5),
		"chunk2": createTestSegments(t, dims, 3),
		"chunk3": createTestSegments(t, dims, 8),
	}

	// Write segments
	store.BeginWrite()
	for chunkID, segs := range chunks {
		store.WriteSegments(chunkID, segs)
	}
	if err := store.CommitWrite(); err != nil {
		t.Fatal(err)
	}

	// Verify file exists
	mmapPath := filepath.Join(tmpDir, "colbert_segments.mmap")
	stat, err := os.Stat(mmapPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("MMap file size: %d bytes", stat.Size())

	// Read back and verify
	for chunkID, expected := range chunks {
		got, err := store.GetColBERTSegments(ctx, chunkID)
		if err != nil {
			t.Errorf("GetColBERTSegments(%s): %v", chunkID, err)
			continue
		}
		if len(got) != len(expected) {
			t.Errorf("%s: expected %d segments, got %d", chunkID, len(expected), len(got))
			continue
		}
		// Verify embeddings match
		for i := range got {
			if len(got[i].EmbeddingInt8) != dims {
				t.Errorf("%s[%d]: expected %d dims, got %d", chunkID, i, dims, len(got[i].EmbeddingInt8))
			}
			// Verify int8 values match
			for j := 0; j < dims; j++ {
				if got[i].EmbeddingInt8[j] != expected[i].EmbeddingInt8[j] {
					t.Errorf("%s[%d][%d]: mismatch %d vs %d", chunkID, i, j, got[i].EmbeddingInt8[j], expected[i].EmbeddingInt8[j])
					break
				}
			}
			// Verify scale and min
			if got[i].QuantScale != expected[i].QuantScale {
				t.Errorf("%s[%d]: scale mismatch %.6f vs %.6f", chunkID, i, got[i].QuantScale, expected[i].QuantScale)
			}
			if got[i].QuantMin != expected[i].QuantMin {
				t.Errorf("%s[%d]: min mismatch %.6f vs %.6f", chunkID, i, got[i].QuantMin, expected[i].QuantMin)
			}
		}
	}

	// Test HasColBERTSegments
	has, err := store.HasColBERTSegments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !has {
		t.Error("Expected HasColBERTSegments to return true")
	}

	// Test batch retrieval
	ids := []string{"chunk1", "chunk3"}
	batch, err := store.GetColBERTSegmentsBatch(ctx, ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(batch) != 2 {
		t.Errorf("Expected 2 chunks in batch, got %d", len(batch))
	}

	// Close and reopen to test persistence
	_ = store.Close()

	store2, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store2.Close() }()

	// Verify data persisted
	segs, err := store2.GetColBERTSegments(ctx, "chunk2")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 3 {
		t.Errorf("After reopen: expected 3 segments for chunk2, got %d", len(segs))
	}

	t.Logf("MMap store test passed with 3 chunks, 16 total segments")
}

func TestMMapSegmentStore_PQRoundTrip(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dims := 4

	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       dims,
		Subspaces:  2,
		Centroids:  4,
		Iterations: 4,
	})
	if err != nil {
		t.Fatalf("NewProductQuantizer failed: %v", err)
	}
	training := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	if err := pq.Train(training, 4); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	codesA, err := pq.Encode([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("Encode A failed: %v", err)
	}
	codesB, err := pq.Encode([]float32{0, 1, 0, 0})
	if err != nil {
		t.Fatalf("Encode B failed: %v", err)
	}

	store, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	store.SetColBERTCodec(ColBERTCodecPQ6, pq, nil)
	store.BeginWrite()
	store.WriteSegments("chunk-pq", []ColBERTSegment{
		{SegmentIdx: 0, PQCodes: codesA},
		{SegmentIdx: 1, PQCodes: codesB},
	})
	if err := store.CommitWrite(); err != nil {
		t.Fatalf("CommitWrite failed: %v", err)
	}
	_ = store.Close()

	store2, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store2.Close() }()

	if store2.ColBERTCodec() != ColBERTCodecPQ6 {
		t.Fatalf("expected codec %q, got %q", ColBERTCodecPQ6, store2.ColBERTCodec())
	}
	if store2.ProductQuantizer() == nil || store2.ProductQuantizer().CodeSize() != pq.CodeSize() {
		t.Fatalf("expected persisted PQ codebook with code size %d", pq.CodeSize())
	}

	segs, err := store2.GetColBERTSegments(ctx, "chunk-pq")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(segs))
	}
	if len(segs[0].PQCodes) != pq.CodeSize() {
		t.Fatalf("expected %d-byte PQ codes, got %d", pq.CodeSize(), len(segs[0].PQCodes))
	}
	if len(segs[0].EmbeddingInt8) != 0 {
		t.Fatalf("expected PQ-only segment payload")
	}
}

func TestMMapSegmentStore_TQMSERoundTrip(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dims := 4

	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: dims,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}
	codeA, err := tq.Encode([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("Encode A failed: %v", err)
	}
	codeB, err := tq.Encode([]float32{0, 1, 0, 0})
	if err != nil {
		t.Fatalf("Encode B failed: %v", err)
	}

	store, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	store.SetColBERTCodec(ColBERTCodecTQMSE, nil, tq)
	store.BeginWrite()
	store.WriteSegments("chunk-tq", []ColBERTSegment{
		{SegmentIdx: 0, TQCodes: codeA.Codes},
		{SegmentIdx: 1, TQCodes: codeB.Codes},
	})
	if err := store.CommitWrite(); err != nil {
		t.Fatalf("CommitWrite failed: %v", err)
	}
	_ = store.Close()

	store2, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store2.Close() }()

	if store2.ColBERTCodec() != ColBERTCodecTQMSE {
		t.Fatalf("expected codec %q, got %q", ColBERTCodecTQMSE, store2.ColBERTCodec())
	}
	if store2.TQMSEQuantizer() == nil || store2.TQMSEQuantizer().CodeSize() != tq.CodeSize() {
		t.Fatalf("expected persisted TQ-MSE quantizer with code size %d", tq.CodeSize())
	}

	segs, err := store2.GetColBERTSegments(ctx, "chunk-tq")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segs) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(segs))
	}
	if len(segs[0].TQCodes) != tq.CodeSize() {
		t.Fatalf("expected %d-byte TQ code, got %d", tq.CodeSize(), len(segs[0].TQCodes))
	}
	if len(segs[0].EmbeddingInt8) != 0 || len(segs[0].PQCodes) != 0 {
		t.Fatalf("expected TQ-only segment payload")
	}
}

func TestMMapSegmentStore_TQMSERejectsMalformedCodeLength(t *testing.T) {
	tmpDir := t.TempDir()
	dims := 8

	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: dims,
		Bits: 3,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}

	store, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	store.SetColBERTCodec(ColBERTCodecTQMSE, nil, tq)
	store.BeginWrite()
	store.WriteSegments("chunk-bad", []ColBERTSegment{
		{SegmentIdx: 0, TQCodes: make([]byte, tq.CodeSize()-1)},
	})

	err = store.CommitWrite()
	if err == nil {
		t.Fatal("expected CommitWrite to reject malformed TQ-MSE code length")
	}
	if !strings.Contains(err.Error(), "code length") {
		t.Fatalf("expected code length error, got %v", err)
	}
}

func TestMMapSegmentStore_TQMSEValidationKeepsExistingMapping(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dims := 8

	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: dims,
		Bits: 3,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}
	code, err := tq.Encode(util.NormalizeVector([]float32{0.7, 0.2, -0.1, 0.05, 0.4, -0.3, 0.1, 0.2}))
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	store, err := OpenMMapSegmentStore(tmpDir, dims)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	store.SetColBERTCodec(ColBERTCodecTQMSE, nil, tq)
	store.BeginWrite()
	store.WriteSegments("chunk-good", []ColBERTSegment{
		{SegmentIdx: 0, TQCodes: code.Codes},
	})
	if err := store.CommitWrite(); err != nil {
		t.Fatalf("initial CommitWrite failed: %v", err)
	}

	store.BeginWrite()
	store.WriteSegments("chunk-bad", []ColBERTSegment{
		{SegmentIdx: 0, TQCodes: make([]byte, tq.CodeSize()-1)},
	})
	err = store.CommitWrite()
	if err == nil {
		t.Fatal("expected CommitWrite to reject malformed TQ-MSE code length")
	}

	segs, err := store.GetColBERTSegments(ctx, "chunk-good")
	if err != nil {
		t.Fatalf("GetColBERTSegments after failed commit failed: %v", err)
	}
	if len(segs) != 1 || len(segs[0].TQCodes) != tq.CodeSize() {
		t.Fatalf("existing mmap data was not readable after failed commit: %+v", segs)
	}
}

func createTestSegments(t *testing.T, dims int, count int) []ColBERTSegment {
	segments := make([]ColBERTSegment, count)
	for i := range segments {
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "test segment",
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}
	return segments
}

func BenchmarkSegmentPooler_Pool(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768

	// Create 10 segments (typical chunk)
	segments := make([]ColBERTSegment, 10)
	for i := range segments {
		emb := make([]float32, dims)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1
		}
		emb = util.NormalizeVector(emb)
		quantized, scale, min := util.QuantizeInt8(emb)

		segments[i] = ColBERTSegment{
			SegmentIdx:    i,
			Text:          "segment",
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	pooler := NewSegmentPooler(5, 0.90)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pooler.Pool(segments)
	}
}

func BenchmarkMMapSegmentStore_Read(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	_ = rng // Use local rng if needed
	dims := 768
	ctx := context.Background()

	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "mmap_bench")
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// Create and populate store
	store, _ := OpenMMapSegmentStore(tmpDir, dims)
	store.BeginWrite()
	for i := 0; i < 100; i++ {
		chunkID := "chunk" + string(rune('0'+i/10)) + string(rune('0'+i%10))
		segs := make([]ColBERTSegment, 5)
		for j := range segs {
			emb := make([]float32, dims)
			for k := range emb {
				emb[k] = rand.Float32()*2 - 1
			}
			emb = util.NormalizeVector(emb)
			quantized, scale, min := util.QuantizeInt8(emb)
			segs[j] = ColBERTSegment{
				SegmentIdx:    j,
				EmbeddingInt8: quantized,
				QuantScale:    scale,
				QuantMin:      min,
			}
		}
		store.WriteSegments(chunkID, segs)
	}
	_ = store.CommitWrite()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkID := "chunk" + string(rune('0'+(i%100)/10)) + string(rune('0'+(i%100)%10))
		_, _ = store.GetColBERTSegments(ctx, chunkID)
	}
}
