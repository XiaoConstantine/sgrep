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

func TestLibSQLStore_ResetIndexClearsArtifacts(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	embedding := makeTestEmbedding(768, 0.5)
	doc := &Document{
		ID:        "test.go:chunk_1",
		FilePath:  "test.go",
		Content:   "func hello() {}",
		StartLine: 1,
		EndLine:   1,
		Embedding: embedding,
	}
	if err := s.Store(ctx, doc); err != nil {
		t.Fatalf("Store failed: %v", err)
	}
	if err := s.StoreFileEmbedding(ctx, &FileEmbedding{
		FilePath:   "test.go",
		Embedding:  embedding,
		ChunkCount: 1,
		TotalLines: 1,
	}); err != nil {
		t.Fatalf("StoreFileEmbedding failed: %v", err)
	}
	if err := s.StoreColBERTSegments(ctx, doc.ID, []ColBERTSegment{{
		SegmentIdx:    0,
		Text:          "func hello",
		Embedding:     embedding,
		EmbeddingInt8: []int8{1, 2, 3},
		QuantScale:    1,
	}}); err != nil {
		t.Fatalf("StoreColBERTSegments failed: %v", err)
	}

	if err := s.ResetIndex(ctx); err != nil {
		t.Fatalf("ResetIndex failed: %v", err)
	}
	stats, err := s.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats failed: %v", err)
	}
	if stats.Documents != 0 || stats.Chunks != 0 {
		t.Fatalf("stats after reset = %+v, want no documents/chunks", stats)
	}
	if s.VectorCount() != 0 {
		t.Fatalf("VectorCount after reset = %d, want 0", s.VectorCount())
	}

	for table, query := range map[string]string{
		"documents_fts":    `SELECT COUNT(*) FROM documents_fts`,
		"file_embeddings":  `SELECT COUNT(*) FROM file_embeddings`,
		"colbert_segments": `SELECT COUNT(*) FROM colbert_segments`,
	} {
		var count int
		if err := s.db.QueryRowContext(ctx, query).Scan(&count); err != nil {
			t.Fatalf("count %s failed: %v", table, err)
		}
		if count != 0 {
			t.Fatalf("%s count = %d, want 0", table, count)
		}
	}
}

func TestOpenForSearchFallsBackWhenTQArtifactInvalid(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "")
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if err := os.WriteFile(TQVectorPath(dir), make([]byte, tqVectorHeaderSize), 0644); err != nil {
		t.Fatalf("write invalid TQ artifact: %v", err)
	}

	opened, err := OpenForSearch(dbPath)
	if err != nil {
		t.Fatalf("OpenForSearch failed: %v", err)
	}
	defer func() { _ = opened.Close() }()
	if _, ok := opened.(*LibSQLStore); !ok {
		t.Fatalf("OpenForSearch returned %T, want *LibSQLStore fallback", opened)
	}
}

func TestOpenForSearchTQPreservesFileEmbeddingSearchWithoutFileTQArtifact(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "")
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	ctx := context.Background()

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	docEmbedding := makeTestEmbedding(768, 0.25)
	doc := &Document{
		ID:        "a.go:chunk_1",
		FilePath:  "a.go",
		Content:   "package main\nfunc main() {}",
		StartLine: 1,
		EndLine:   2,
		Embedding: docEmbedding,
	}
	if err := s.Store(ctx, doc); err != nil {
		t.Fatalf("Store failed: %v", err)
	}
	if err := s.StoreFileEmbedding(ctx, &FileEmbedding{
		FilePath:   "a.go",
		Embedding:  docEmbedding,
		ChunkCount: 1,
		TotalLines: 2,
	}); err != nil {
		t.Fatalf("StoreFileEmbedding failed: %v", err)
	}
	if _, err := BuildTQVectorStore(ctx, dir, []string{doc.ID}, [][]float32{docEmbedding}, TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42}); err != nil {
		t.Fatalf("BuildTQVectorStore failed: %v", err)
	}
	if err := s.Checkpoint(ctx); err != nil {
		t.Fatalf("Checkpoint failed: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	opened, err := OpenForSearch(dbPath)
	if err != nil {
		t.Fatalf("OpenForSearch failed: %v", err)
	}
	defer func() { _ = opened.Close() }()
	fileStore, ok := opened.(FileEmbeddingStorer)
	if !ok {
		t.Fatalf("OpenForSearch returned %T without FileEmbeddingStorer", opened)
	}

	paths, distances, err := fileStore.SearchFileEmbeddings(ctx, docEmbedding, 1, 2)
	if err != nil {
		t.Fatalf("SearchFileEmbeddings failed: %v", err)
	}
	if len(paths) != 1 || paths[0] != "a.go" {
		t.Fatalf("SearchFileEmbeddings paths=%v distances=%v, want a.go", paths, distances)
	}
	chunks, err := fileStore.GetChunksByFilePath(ctx, "a.go")
	if err != nil {
		t.Fatalf("GetChunksByFilePath failed: %v", err)
	}
	if len(chunks) != 1 || chunks[0].ID != doc.ID {
		t.Fatalf("GetChunksByFilePath chunks=%v, want %s", chunks, doc.ID)
	}
}

func TestOpenForSearchTQUsesFileVectorArtifact(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "")
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	ctx := context.Background()

	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	docEmbedding := makeTestEmbedding(768, 0.35)
	doc := &Document{
		ID:        "a.go:chunk_1",
		FilePath:  "a.go",
		Content:   "package main\nfunc main() {}",
		StartLine: 1,
		EndLine:   2,
		Embedding: docEmbedding,
	}
	if err := s.Store(ctx, doc); err != nil {
		t.Fatalf("Store failed: %v", err)
	}
	if err := s.StoreFileEmbedding(ctx, &FileEmbedding{
		FilePath:   "a.go",
		Embedding:  docEmbedding,
		ChunkCount: 1,
		TotalLines: 2,
	}); err != nil {
		t.Fatalf("StoreFileEmbedding failed: %v", err)
	}
	if _, err := BuildTQVectorStore(ctx, dir, []string{doc.ID}, [][]float32{docEmbedding}, TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42}); err != nil {
		t.Fatalf("BuildTQVectorStore failed: %v", err)
	}
	if _, err := BuildTQFileVectorStore(ctx, dir, []string{doc.FilePath}, [][]float32{docEmbedding}, TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42}); err != nil {
		t.Fatalf("BuildTQFileVectorStore failed: %v", err)
	}
	if err := s.Checkpoint(ctx); err != nil {
		t.Fatalf("Checkpoint failed: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	opened, err := OpenForSearch(dbPath)
	if err != nil {
		t.Fatalf("OpenForSearch failed: %v", err)
	}
	defer func() { _ = opened.Close() }()
	tq, ok := opened.(*TQSearchStore)
	if !ok {
		t.Fatalf("OpenForSearch returned %T, want *TQSearchStore", opened)
	}
	if tq.fileDense == nil {
		t.Fatal("TQSearchStore did not open file vector artifact")
	}
	if tq.fileStore != nil {
		t.Fatal("TQSearchStore opened libSQL file embedding delegate despite file vector artifact")
	}

	paths, distances, err := tq.SearchFileEmbeddings(ctx, docEmbedding, 1, 2)
	if err != nil {
		t.Fatalf("SearchFileEmbeddings failed: %v", err)
	}
	if len(paths) != 1 || paths[0] != "a.go" {
		t.Fatalf("SearchFileEmbeddings paths=%v distances=%v, want a.go", paths, distances)
	}
	chunks, err := tq.GetChunksByFilePath(ctx, "a.go")
	if err != nil {
		t.Fatalf("GetChunksByFilePath failed: %v", err)
	}
	if len(chunks) != 1 || chunks[0].ID != doc.ID {
		t.Fatalf("GetChunksByFilePath chunks=%v, want %s", chunks, doc.ID)
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

func TestLibSQLStore_PruneIndexPreservesLiveColBERTSegments(t *testing.T) {
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
	if err := s.StoreFileEmbedding(ctx, &FileEmbedding{
		FilePath:   "file2.go",
		Embedding:  makeTestEmbedding(768, 0.25),
		ChunkCount: 1,
		TotalLines: 1,
	}); err != nil {
		t.Fatalf("StoreFileEmbedding failed: %v", err)
	}
	if err := s.SaveColBERTMetadata(ctx, ColBERTCodecInt8, nil, nil); err != nil {
		t.Fatalf("SaveColBERTMetadata failed: %v", err)
	}
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"file1.go:chunk_1": {{SegmentIdx: 0, Text: "first", Embedding: makeTestEmbedding(768, 0.3)}},
		"file2.go:chunk_1": {{SegmentIdx: 0, Text: "second", Embedding: makeTestEmbedding(768, 0.4)}},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	if err := s.PruneIndex(ctx, []string{"file1.go:chunk_1"}, []string{"file1.go"}); err != nil {
		t.Fatalf("PruneIndex failed: %v", err)
	}

	liveSegments, err := s.GetColBERTSegments(ctx, "file1.go:chunk_1")
	if err != nil {
		t.Fatalf("GetColBERTSegments(live) failed: %v", err)
	}
	if len(liveSegments) != 1 {
		t.Fatalf("expected live ColBERT segment to remain, got %d", len(liveSegments))
	}
	staleSegments, err := s.GetColBERTSegments(ctx, "file2.go:chunk_1")
	if err != nil {
		t.Fatalf("GetColBERTSegments(stale) failed: %v", err)
	}
	if len(staleSegments) != 0 {
		t.Fatalf("expected stale ColBERT segment to be pruned, got %d", len(staleSegments))
	}

	var staleDocs int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM documents WHERE filepath = ?`, "file2.go").Scan(&staleDocs); err != nil {
		t.Fatalf("query stale docs failed: %v", err)
	}
	if staleDocs != 0 {
		t.Fatalf("expected stale document to be pruned, got %d", staleDocs)
	}
	var staleFiles int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM file_embeddings WHERE filepath = ?`, "file2.go").Scan(&staleFiles); err != nil {
		t.Fatalf("query stale file embeddings failed: %v", err)
	}
	if staleFiles != 0 {
		t.Fatalf("expected stale file embedding to be pruned, got %d", staleFiles)
	}
	if s.ColBERTCodec() != ColBERTCodecInt8 {
		t.Fatalf("PruneIndex reset ColBERT metadata, got codec %q", s.ColBERTCodec())
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

func TestLibSQLStore_FTSIndexesEnrichedLexicalMetadata(t *testing.T) {
	s, err := OpenLibSQL(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	doc := &Document{
		ID: "auth.go:chunk_1", FilePath: "auth.go", Content: "func run() {}", StartLine: 1, EndLine: 1,
		Embedding: makeTestEmbedding(768, 0.1), Metadata: map[string]string{"lexical": "jwt validator refresh token"},
	}
	if err := s.Store(context.Background(), doc); err != nil {
		t.Fatal(err)
	}
	hits, err := s.BM25Search(context.Background(), "jwt", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].ID != doc.ID {
		t.Fatalf("BM25Search enriched metadata = %+v", hits)
	}
}

func TestLibSQLStore_FTSUpdateRemovesOldTerms(t *testing.T) {
	s, err := OpenLibSQL(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	ctx := context.Background()
	doc := &Document{ID: "same", FilePath: "old.go", Content: "obsoleteTerm", StartLine: 1, EndLine: 1, Embedding: makeTestEmbedding(768, 0.1)}
	if err := s.Store(ctx, doc); err != nil {
		t.Fatal(err)
	}
	doc.FilePath = "new.go"
	doc.Content = "replacementTerm"
	if err := s.Store(ctx, doc); err != nil {
		t.Fatal(err)
	}
	oldHits, err := s.BM25Search(ctx, "obsoleteTerm", 10)
	if err != nil {
		t.Fatal(err)
	}
	newHits, err := s.BM25Search(ctx, "replacementTerm", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(oldHits) != 0 || len(newHits) != 1 || newHits[0].ID != doc.ID {
		t.Fatalf("updated FTS old=%+v new=%+v", oldHits, newHits)
	}
}

func TestLibSQLStore_CloseMakesMetadataVisibleToImmutableSearch(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "index.db")
	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	doc := &Document{
		ID: "visible.go:chunk_1", FilePath: "visible.go", Content: "func visible() {}",
		StartLine: 1, EndLine: 1,
	}
	if err := s.StoreMetadataBatch(ctx, []*Document{doc}); err != nil {
		_ = s.Close()
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	metadata, err := OpenSQLiteMetadata(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = metadata.Close() }()
	got, err := metadata.GetChunksByFilePath(ctx, doc.FilePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != doc.ID || got[0].Content != doc.Content {
		t.Fatalf("immutable metadata after Close = %+v, want %s", got, doc.ID)
	}
}

func TestLibSQLStore_GetChunksForColBERT_IncludesCompactMetadataRows(t *testing.T) {
	dir := t.TempDir()
	s, err := OpenLibSQL(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	doc := &Document{
		ID: "compact.go:chunk_1", FilePath: "compact.go", Content: "func compact() {}",
		StartLine: 1, EndLine: 1, Metadata: map[string]string{"description": "compact helper"},
	}
	if err := s.StoreMetadataBatch(ctx, []*Document{doc}); err != nil {
		t.Fatal(err)
	}

	chunks, err := s.GetChunksForColBERT(ctx, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].ID != doc.ID {
		t.Fatalf("compact metadata chunks = %+v, want %s", chunks, doc.ID)
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
	if err := s.SaveColBERTMetadata(ctx, ColBERTCodecPQ6, pq, nil); err != nil {
		t.Fatalf("SaveColBERTMetadata failed: %v", err)
	}
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"chunk-1": {
			{SegmentIdx: 0, Text: "seg", PQCodes: codes},
		},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	gotCodec, gotPQ, gotTQ, err := s.LoadColBERTMetadata(ctx)
	if err != nil {
		t.Fatalf("LoadColBERTMetadata failed: %v", err)
	}
	if gotCodec != ColBERTCodecPQ6 {
		t.Fatalf("expected codec %q, got %q", ColBERTCodecPQ6, gotCodec)
	}
	if gotPQ == nil || gotPQ.CodeSize() != pq.CodeSize() {
		t.Fatalf("expected persisted PQ codebook with code size %d", pq.CodeSize())
	}
	if gotTQ != nil {
		t.Fatal("expected nil TQ-MSE quantizer for PQ codec")
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

func TestLibSQLStore_ColBERTTQMSEMetadataAndSegments(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	t.Setenv("SGREP_DIMS", "4")
	s, err := OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: 4,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}
	code, err := tq.Encode([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	ctx := context.Background()
	if err := s.SaveColBERTMetadata(ctx, ColBERTCodecTQMSE, nil, tq); err != nil {
		t.Fatalf("SaveColBERTMetadata failed: %v", err)
	}
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"chunk-1": {
			{SegmentIdx: 0, Text: "seg", TQCodes: code.Codes},
		},
	}); err != nil {
		t.Fatalf("StoreColBERTSegmentsBatch failed: %v", err)
	}

	gotCodec, gotPQ, gotTQ, err := s.LoadColBERTMetadata(ctx)
	if err != nil {
		t.Fatalf("LoadColBERTMetadata failed: %v", err)
	}
	if gotCodec != ColBERTCodecTQMSE {
		t.Fatalf("expected codec %q, got %q", ColBERTCodecTQMSE, gotCodec)
	}
	if gotPQ != nil {
		t.Fatal("expected nil PQ codebook for TQ-MSE codec")
	}
	if gotTQ == nil || gotTQ.CodeSize() != tq.CodeSize() {
		t.Fatalf("expected persisted TQ-MSE quantizer with code size %d", tq.CodeSize())
	}

	segs, err := s.GetColBERTSegments(ctx, "chunk-1")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segs) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segs))
	}
	if len(segs[0].TQCodes) != len(code.Codes) {
		t.Fatalf("expected %d TQ codes, got %d", len(code.Codes), len(segs[0].TQCodes))
	}
	if len(segs[0].EmbeddingInt8) != 0 || len(segs[0].PQCodes) != 0 {
		t.Fatalf("expected TQ segment without int8/PQ payload")
	}
}

func TestLibSQLStore_ExportColBERTSegmentsStreamsCompactOrderedChunks(t *testing.T) {
	s, err := OpenLibSQL(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	ctx := context.Background()
	if err := s.StoreColBERTSegmentsBatch(ctx, map[string][]ColBERTSegment{
		"chunk-b": {
			{SegmentIdx: 1, Text: "not exported", EmbeddingInt8: []int8{2, 3}, QuantScale: 0.5, QuantMin: -1},
			{SegmentIdx: 0, Text: "not exported", EmbeddingInt8: []int8{0, 1}, QuantScale: 0.25, QuantMin: 0},
		},
		"chunk-a": {{SegmentIdx: 0, Text: "not exported", PQCodes: []byte{4, 5}}},
		"chunk-c": {{SegmentIdx: 0, Text: "not exported", TQCodes: []byte{6, 7}}},
	}); err != nil {
		t.Fatal(err)
	}

	var chunkIDs []string
	var exported [][]ColBERTSegment
	err = s.ExportColBERTSegments(ctx, func(chunkID string, segments []ColBERTSegment) error {
		chunkIDs = append(chunkIDs, chunkID)
		exported = append(exported, segments)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunkIDs) != 3 || chunkIDs[0] != "chunk-a" || chunkIDs[1] != "chunk-b" || chunkIDs[2] != "chunk-c" {
		t.Fatalf("chunk order = %v", chunkIDs)
	}
	if len(exported[1]) != 2 || exported[1][0].SegmentIdx != 0 || exported[1][1].SegmentIdx != 1 {
		t.Fatalf("segment order = %+v", exported[1])
	}
	if exported[1][0].Text != "" || exported[1][0].QuantMin != 0 || exported[1][1].QuantMin != -1 {
		t.Fatalf("int8 export = %+v", exported[1])
	}
	if len(exported[0][0].PQCodes) != 2 || len(exported[2][0].TQCodes) != 2 {
		t.Fatalf("compact codecs not exported: pq=%v tq=%v", exported[0][0].PQCodes, exported[2][0].TQCodes)
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

func BenchmarkLibSQLStore_ExportColBERTSegments14K(b *testing.B) {
	const (
		chunkCount       = 14126
		segmentsPerChunk = 4
	)

	s, err := OpenLibSQL(filepath.Join(b.TempDir(), "test.db"))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = s.Close() })

	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		b.Fatal(err)
	}
	docStmt, err := tx.PrepareContext(ctx, `INSERT INTO documents (id, filepath, content, start_line, end_line, metadata, is_test) VALUES (?, ?, ?, 1, 1, '{}', 0)`)
	if err != nil {
		b.Fatal(err)
	}
	segmentStmt, err := tx.PrepareContext(ctx, `INSERT INTO colbert_segments (chunk_id, segment_idx, segment_text, tq_codes) VALUES (?, ?, '', ?)`)
	if err != nil {
		b.Fatal(err)
	}
	content := string(make([]byte, 1024))
	code := make([]byte, 384)
	for i := range chunkCount {
		chunkID := "file.go:chunk_" + itoa(i)
		if _, err := docStmt.ExecContext(ctx, chunkID, "file.go", content); err != nil {
			b.Fatal(err)
		}
		for j := range segmentsPerChunk {
			if _, err := segmentStmt.ExecContext(ctx, chunkID, j, code); err != nil {
				b.Fatal(err)
			}
		}
	}
	_ = docStmt.Close()
	_ = segmentStmt.Close()
	if err := tx.Commit(); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.SetBytes(chunkCount * segmentsPerChunk * int64(len(code)))
	b.ResetTimer()
	for b.Loop() {
		totalSegments := 0
		err := s.ExportColBERTSegments(ctx, func(_ string, segments []ColBERTSegment) error {
			totalSegments += len(segments)
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		if totalSegments != chunkCount*segmentsPerChunk {
			b.Fatalf("exported %d segments, want %d", totalSegments, chunkCount*segmentsPerChunk)
		}
	}
}
