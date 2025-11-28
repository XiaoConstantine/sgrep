package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestBufferedStore_Basic(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}
	defer s.Close()

	// Store a document
	ctx := context.Background()
	doc := &Document{
		ID:        "test-1",
		FilePath:  "test.go",
		Content:   "func hello() {}",
		StartLine: 1,
		EndLine:   1,
		Embedding: makeTestEmbedding(768, 0.5),
	}

	if err := s.Store(ctx, doc); err != nil {
		t.Fatalf("Store failed: %v", err)
	}

	// Flush to ensure data is persisted
	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	// Search should find it
	docs, distances, err := s.Search(ctx, makeTestEmbedding(768, 0.5), 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(docs) != 1 {
		t.Errorf("expected 1 result, got %d", len(docs))
	}
	if len(docs) > 0 && docs[0].ID != "test-1" {
		t.Errorf("expected doc ID 'test-1', got %s", docs[0].ID)
	}
	if len(distances) > 0 && distances[0] > 0.1 {
		t.Errorf("expected small distance, got %f", distances[0])
	}
}

func TestBufferedStore_BatchFlush(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Store many documents (more than vec0ChunkSize)
	docs := make([]*Document, 100)
	for i := 0; i < 100; i++ {
		docs[i] = &Document{
			ID:        "doc-" + string(rune('a'+i%26)) + string(rune('0'+i/26)),
			FilePath:  "test.go",
			Content:   "func test() {}",
			StartLine: i,
			EndLine:   i + 1,
			Embedding: makeTestEmbedding(768, float32(i)/100.0),
		}
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	// Flush remaining
	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	// Check count
	if s.VectorCount() != 100 {
		t.Errorf("expected 100 vectors, got %d", s.VectorCount())
	}

	// Search should work
	results, _, err := s.Search(ctx, makeTestEmbedding(768, 0.5), 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results, got none")
	}
}

func TestBufferedStore_SearchModes(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	// Test in-memory mode (default for small datasets)
	s, err := OpenBuffered(dbPath, WithSearchMode(searchModeInMemory))
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}

	if s.SearchMode() != "in-memory" {
		t.Errorf("expected in-memory mode, got %s", s.SearchMode())
	}
	s.Close()

	// Test sqlite mode
	s2, err := OpenBuffered(dbPath, WithSearchMode(searchModeSQLite))
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}

	if s2.SearchMode() != "sqlite-vec" {
		t.Errorf("expected sqlite-vec mode, got %s", s2.SearchMode())
	}
	s2.Close()
}

func TestBufferedStore_Quantization(t *testing.T) {
	testCases := []struct {
		name string
		mode QuantizationMode
	}{
		{"none", QuantizeNone},
		{"int8", QuantizeInt8},
		// Binary quantization with Hamming distance requires special vec0 syntax
		// that may not be available in all sqlite-vec versions
		// {"binary", QuantizeBinary},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			dbPath := filepath.Join(dir, "test.db")

			s, err := OpenBuffered(dbPath, WithBufferedQuantization(tc.mode))
			if err != nil {
				t.Fatalf("OpenBuffered failed: %v", err)
			}
			defer s.Close()

			ctx := context.Background()
			doc := &Document{
				ID:        "test-1",
				FilePath:  "test.go",
				Content:   "func hello() {}",
				StartLine: 1,
				EndLine:   1,
				Embedding: makeTestEmbedding(768, 0.5),
			}

			if err := s.Store(ctx, doc); err != nil {
				t.Fatalf("Store failed: %v", err)
			}

			if err := s.Flush(ctx); err != nil {
				t.Fatalf("Flush failed: %v", err)
			}

			// Search should work
			docs, _, err := s.Search(ctx, makeTestEmbedding(768, 0.5), 10, 2.0)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(docs) != 1 {
				t.Errorf("expected 1 result, got %d", len(docs))
			}
		})
	}
}

func TestBufferedStore_DeleteByPath(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Store documents
	docs := []*Document{
		{ID: "doc-1", FilePath: "file1.go", Content: "func a() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.1)},
		{ID: "doc-2", FilePath: "file1.go", Content: "func b() {}", StartLine: 2, EndLine: 2, Embedding: makeTestEmbedding(768, 0.2)},
		{ID: "doc-3", FilePath: "file2.go", Content: "func c() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.3)},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	// Delete file1.go
	if err := s.DeleteByPath(ctx, "file1.go"); err != nil {
		t.Fatalf("DeleteByPath failed: %v", err)
	}

	// Should have 1 vector left
	if s.VectorCount() != 1 {
		t.Errorf("expected 1 vector, got %d", s.VectorCount())
	}
}

func TestBufferedStore_Stats(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Store documents
	docs := []*Document{
		{ID: "doc-1", FilePath: "file1.go", Content: "func a() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.1)},
		{ID: "doc-2", FilePath: "file1.go", Content: "func b() {}", StartLine: 2, EndLine: 2, Embedding: makeTestEmbedding(768, 0.2)},
		{ID: "doc-3", FilePath: "file2.go", Content: "func c() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.3)},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	stats, err := s.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}

	if stats.Documents != 2 { // 2 unique files
		t.Errorf("expected 2 documents, got %d", stats.Documents)
	}

	if stats.Chunks != 3 { // 3 chunks
		t.Errorf("expected 3 chunks, got %d", stats.Chunks)
	}

	// Size should be > 0
	if stats.SizeBytes == 0 {
		t.Error("expected non-zero size")
	}
}

func TestBufferedStore_Persistence(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	// Create store and add data
	s, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}

	ctx := context.Background()
	doc := &Document{
		ID:        "persistent-doc",
		FilePath:  "test.go",
		Content:   "func hello() {}",
		StartLine: 1,
		EndLine:   1,
		Embedding: makeTestEmbedding(768, 0.5),
	}

	if err := s.Store(ctx, doc); err != nil {
		t.Fatalf("Store failed: %v", err)
	}

	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	s.Close()

	// Reopen and verify data persisted
	s2, err := OpenBuffered(dbPath)
	if err != nil {
		t.Fatalf("Reopen failed: %v", err)
	}
	defer s2.Close()

	if s2.VectorCount() != 1 {
		t.Errorf("expected 1 vector after reopen, got %d", s2.VectorCount())
	}

	// Search should work
	docs, _, err := s2.Search(ctx, makeTestEmbedding(768, 0.5), 10, 2.0)
	if err != nil {
		t.Fatalf("Search after reopen failed: %v", err)
	}

	if len(docs) != 1 {
		t.Errorf("expected 1 result, got %d", len(docs))
	}
}

// makeTestEmbedding creates a test embedding with consistent values
func makeTestEmbedding(dims int, value float32) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = value
	}
	return vec
}

func TestBufferedStore_LargeChunkBatch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large batch test in short mode")
	}

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenBuffered(dbPath, WithBufferedQuantization(QuantizeInt8))
	if err != nil {
		t.Fatalf("OpenBuffered failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Store a reasonable batch of documents
	batchSize := 100
	docs := make([]*Document, batchSize)
	for i := 0; i < batchSize; i++ {
		docs[i] = &Document{
			ID:        fmt.Sprintf("chunk-%d", i),
			FilePath:  "test.go",
			Content:   "func test() {}",
			StartLine: i,
			EndLine:   i + 1,
			Embedding: makeTestEmbedding(768, float32(i)/float32(batchSize)),
		}
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if err := s.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	if s.VectorCount() != int64(batchSize) {
		t.Errorf("expected %d vectors, got %d", batchSize, s.VectorCount())
	}

	// Note: vec0 allocates fixed-size chunks per INSERT (not per transaction).
	// This is a known limitation of sqlite-vec's chunk architecture.
	// See: https://github.com/asg017/sqlite-vec/issues/185
	// The BufferedStore still provides value by:
	// 1. Batching document metadata inserts
	// 2. Adaptive search mode (in-memory for small, sqlite-vec for large)
	// 3. Flush semantics for reliability
	info, err := os.Stat(dbPath)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}

	t.Logf("Stored %d vectors in %d bytes (%.2f KB/vector)",
		batchSize, info.Size(), float64(info.Size())/float64(batchSize)/1024)
}
