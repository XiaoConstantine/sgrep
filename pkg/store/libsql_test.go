//go:build !sqlite_vec
// +build !sqlite_vec

package store

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

func TestLibSQLStore_Basic(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

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

	if s.VectorCount() != 1 {
		t.Errorf("expected 1 vector, got %d", s.VectorCount())
	}
}

func TestLibSQLStore_StoreBatch(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	docs := make([]*Document, 100)
	for i := 0; i < 100; i++ {
		docs[i] = &Document{
			ID:        "doc-" + itoa(i),
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

	if s.VectorCount() != 100 {
		t.Errorf("expected 100 vectors, got %d", s.VectorCount())
	}
}

func TestLibSQLStore_Search(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()

	// Store documents with varying embeddings
	docs := make([]*Document, 50)
	for i := 0; i < 50; i++ {
		docs[i] = &Document{
			ID:        "doc-" + itoa(i),
			FilePath:  "test.go",
			Content:   "func test" + itoa(i) + "() {}",
			StartLine: i,
			EndLine:   i + 1,
			Embedding: makeTestEmbedding(768, float32(i)/50.0),
		}
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	// Search for similar vectors
	query := makeTestEmbedding(768, 0.5)
	results, distances, err := s.Search(ctx, query, 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results, got none")
	}

	// Verify distances are sorted
	for i := 1; i < len(distances); i++ {
		if distances[i] < distances[i-1] {
			t.Errorf("distances not sorted: %f < %f", distances[i], distances[i-1])
		}
	}
}

func TestLibSQLStore_SearchEmpty(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	query := makeTestEmbedding(768, 0.5)
	results, distances, err := s.Search(ctx, query, 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 0 || len(distances) != 0 {
		t.Error("expected empty results for empty store")
	}
}

func TestLibSQLStore_DeleteByPath(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()

	docs := []*Document{
		{ID: "doc-1", FilePath: "file1.go", Content: "func a() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.1)},
		{ID: "doc-2", FilePath: "file1.go", Content: "func b() {}", StartLine: 2, EndLine: 2, Embedding: makeTestEmbedding(768, 0.2)},
		{ID: "doc-3", FilePath: "file2.go", Content: "func c() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.3)},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if err := s.DeleteByPath(ctx, "file1.go"); err != nil {
		t.Fatalf("DeleteByPath failed: %v", err)
	}

	if s.VectorCount() != 1 {
		t.Errorf("expected 1 vector after delete, got %d", s.VectorCount())
	}
}

func TestLibSQLStore_DeleteByPath_RemovesLateInteractionArtifacts(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	docs := []*Document{
		{
			ID:        "file1.go:chunk_1",
			FilePath:  "file1.go",
			Content:   "func first() {}",
			StartLine: 1,
			EndLine:   1,
			Embedding: makeTestEmbedding(768, 0.1),
			Metadata:  map[string]string{"description": "func first in file1.go"},
		},
		{
			ID:        "file2.go:chunk_1",
			FilePath:  "file2.go",
			Content:   "func second() {}",
			StartLine: 1,
			EndLine:   1,
			Embedding: makeTestEmbedding(768, 0.2),
			Metadata:  map[string]string{"description": "func second in file2.go"},
		},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if err := s.StoreFileEmbedding(ctx, &FileEmbedding{
		FilePath:   "file1.go",
		Embedding:  makeTestEmbedding(768, 0.15),
		ChunkCount: 1,
		TotalLines: 1,
	}); err != nil {
		t.Fatalf("StoreFileEmbedding failed: %v", err)
	}

	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"file1.go:chunk_1": {
			{
				SegmentIdx: 0,
				Text:       "func first in file1.go",
				Embedding:  makeTestEmbedding(768, 0.3),
			},
		},
		"file2.go:chunk_1": {
			{
				SegmentIdx: 0,
				Text:       "func second in file2.go",
				Embedding:  makeTestEmbedding(768, 0.4),
			},
		},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	if err := s.DeleteByPath(ctx, "file1.go"); err != nil {
		t.Fatalf("DeleteByPath failed: %v", err)
	}

	var fileEmbeddingCount int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM file_embeddings WHERE filepath = ?`, "file1.go").Scan(&fileEmbeddingCount); err != nil {
		t.Fatalf("query file_embeddings failed: %v", err)
	}
	if fileEmbeddingCount != 0 {
		t.Fatalf("expected file embedding to be deleted, got %d rows", fileEmbeddingCount)
	}

	var segmentCount int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM colbert_segments WHERE chunk_id LIKE ?`, "file1.go:%").Scan(&segmentCount); err != nil {
		t.Fatalf("query colbert_segments failed: %v", err)
	}
	if segmentCount != 0 {
		t.Fatalf("expected ColBERT segments to be deleted, got %d rows", segmentCount)
	}

	remainingSegments, err := s.GetColBERTSegments(ctx, "file2.go:chunk_1")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(remainingSegments) != 1 {
		t.Fatalf("expected unrelated ColBERT segments to remain, got %d", len(remainingSegments))
	}
}

func TestLibSQLStore_GetChunksForColBERT_LoadsDescriptionViaJSONExtract(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	docs := []*Document{
		{
			ID:        "file1.go:chunk_1",
			FilePath:  "file1.go",
			Content:   "func first() {}",
			StartLine: 1,
			EndLine:   1,
			Embedding: makeTestEmbedding(768, 0.1),
			Metadata:  map[string]string{"description": "first helper"},
		},
		{
			ID:        "file2.go:chunk_1",
			FilePath:  "file2.go",
			Content:   "func second() {}",
			StartLine: 1,
			EndLine:   1,
			Embedding: makeTestEmbedding(768, 0.2),
		},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	chunks, err := s.GetChunksForColBERT(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetChunksForColBERT failed: %v", err)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	if chunks[0].Description != "first helper" {
		t.Fatalf("expected first description %q, got %q", "first helper", chunks[0].Description)
	}

	if chunks[1].Description != "" {
		t.Fatalf("expected second description to be empty, got %q", chunks[1].Description)
	}
}

func TestLibSQLStore_ColBERTPQMetadataAndSegments(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       4,
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
	codes, err := pq.Encode([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	ctx := context.Background()
	if err := s.SaveColBERTMetadata(ctx, ColBERTCodecPQ6, pq); err != nil {
		t.Fatalf("SaveColBERTMetadata failed: %v", err)
	}
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"chunk-1": {
			{SegmentIdx: 0, Text: "seg", PQCodes: codes},
		},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	gotCodec, gotPQ, err := s.LoadColBERTMetadata(ctx)
	if err != nil {
		t.Fatalf("LoadColBERTMetadata failed: %v", err)
	}
	if gotCodec != ColBERTCodecPQ6 {
		t.Fatalf("expected codec %q, got %q", ColBERTCodecPQ6, gotCodec)
	}
	if gotPQ == nil || gotPQ.CodeSize() != pq.CodeSize() {
		t.Fatalf("expected persisted PQ codebook with code size %d", pq.CodeSize())
	}

	segs, err := s.GetColBERTSegments(ctx, "chunk-1")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segs) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segs))
	}
	if len(segs[0].PQCodes) != len(codes) {
		t.Fatalf("expected %d PQ codes, got %d", len(codes), len(segs[0].PQCodes))
	}
	if len(segs[0].EmbeddingInt8) != 0 {
		t.Fatalf("expected PQ segment without int8 payload, got %d bytes", len(segs[0].EmbeddingInt8))
	}
}

func TestLibSQLStore_ColBERTInt8ZeroQuantMinRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	seg := ColBERTSegment{
		SegmentIdx:    0,
		Text:          "zero-min",
		EmbeddingInt8: []int8{0, 1, 2, 3},
		QuantScale:    0.25,
		QuantMin:      0,
	}
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"chunk-zero": {seg},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	segs, err := s.GetColBERTSegments(ctx, "chunk-zero")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segs) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segs))
	}
	if segs[0].QuantMin != 0 {
		t.Fatalf("expected QuantMin 0, got %f", segs[0].QuantMin)
	}
	if segs[0].QuantScale != seg.QuantScale {
		t.Fatalf("expected QuantScale %f, got %f", seg.QuantScale, segs[0].QuantScale)
	}
}

func TestLibSQLStore_Stats(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()

	docs := []*Document{
		{ID: "doc-1", FilePath: "file1.go", Content: "func a() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.1)},
		{ID: "doc-2", FilePath: "file1.go", Content: "func b() {}", StartLine: 2, EndLine: 2, Embedding: makeTestEmbedding(768, 0.2)},
		{ID: "doc-3", FilePath: "file2.go", Content: "func c() {}", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.3)},
	}

	if err := s.StoreBatch(ctx, docs); err != nil {
		t.Fatalf("StoreBatch failed: %v", err)
	}

	stats, err := s.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}

	if stats.Documents != 2 {
		t.Errorf("expected 2 unique files, got %d", stats.Documents)
	}

	if stats.Chunks != 3 {
		t.Errorf("expected 3 chunks, got %d", stats.Chunks)
	}

	if stats.SizeBytes == 0 {
		t.Error("expected non-zero size")
	}
}

func TestLibSQLStore_Persistence(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	// Create and populate store
	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
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
	if err := s.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Small delay to ensure file is released
	time.Sleep(100 * time.Millisecond)

	// Reopen and verify
	s2, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("Reopen failed: %v", err)
	}
	defer func() { _ = s2.Close() }()

	if s2.VectorCount() != 1 {
		t.Errorf("expected 1 vector after reopen, got %d", s2.VectorCount())
	}
}

func TestLibSQLStore_Quantization(t *testing.T) {
	testCases := []struct {
		name string
		mode QuantizationMode
	}{
		{"none", QuantizeNone},
		{"int8", QuantizeInt8},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			dbPath := filepath.Join(dir, "test.db")

			s, err := OpenLibSQL(dbPath, WithLibSQLQuantization(tc.mode))
			if err != nil {
				t.Fatalf("OpenLibSQL failed: %v", err)
			}
			defer func() { _ = s.Close() }()

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

			if s.VectorCount() != 1 {
				t.Errorf("expected 1 vector, got %d", s.VectorCount())
			}
		})
	}
}

func TestParseVectorString(t *testing.T) {
	cases := []struct {
		input string
		want  []float32
	}{
		{"[1,2,3]", []float32{1, 2, 3}},
		{"[1.5, 2.5, 3.5]", []float32{1.5, 2.5, 3.5}},
		{"[]", nil},
		{"", nil},
	}

	for _, c := range cases {
		got := parseVectorString(c.input)
		if c.want == nil {
			if got != nil {
				t.Errorf("parseVectorString(%q) = %v, want nil", c.input, got)
			}
			continue
		}
		if len(got) != len(c.want) {
			t.Errorf("parseVectorString(%q) len = %d, want %d", c.input, len(got), len(c.want))
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("parseVectorString(%q)[%d] = %f, want %f", c.input, i, got[i], c.want[i])
			}
		}
	}
}

func TestFormatVectorString(t *testing.T) {
	vec := []float32{1.0, 2.5, 3.0}
	got := formatVectorString(vec)
	expected := "[1,2.5,3]"
	if got != expected {
		t.Errorf("formatVectorString(%v) = %q, want %q", vec, got, expected)
	}
}

// Benchmark to compare disk size
func BenchmarkLibSQLStore_DiskSize(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping disk size benchmark in short mode")
	}

	for i := 0; i < b.N; i++ {
		dir := b.TempDir()
		dbPath := filepath.Join(dir, "test.db")

		s, err := OpenLibSQL(dbPath)
		if err != nil {
			b.Fatalf("OpenLibSQL failed: %v", err)
		}

		ctx := context.Background()
		docs := make([]*Document, 1000)
		for j := 0; j < 1000; j++ {
			docs[j] = &Document{
				ID:        "doc-" + itoa(j),
				FilePath:  "test.go",
				Content:   "func test() { /* some content */ }",
				StartLine: j,
				EndLine:   j + 1,
				Embedding: makeTestEmbedding(768, float32(j)/1000.0),
			}
		}

		if err := s.StoreBatch(ctx, docs); err != nil {
			b.Fatalf("StoreBatch failed: %v", err)
		}

		_ = s.Close()

		info, err := os.Stat(dbPath)
		if err != nil {
			b.Fatalf("Stat failed: %v", err)
		}

		b.Logf("LibSQL: %d vectors in %d bytes (%.2f KB/vector)",
			1000, info.Size(), float64(info.Size())/1000.0/1024.0)
	}
}
