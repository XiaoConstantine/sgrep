//go:build !sqlite_vec
// +build !sqlite_vec

package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
)

func TestSearcherDocumentLevelUsesTQFileVectorArtifact(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "tqmse")

	ctx := context.Background()
	queryEmbedding := searchTestVector(768, 1)
	opened := openSearchTestTQStore(t, ctx, queryEmbedding, true, false)
	defer func() { _ = opened.Close() }()

	searcher := NewWithConfig(Config{
		Store:    opened,
		Embedder: fixedSearchTestEmbedder(t, queryEmbedding),
	})
	opts := DefaultSearchOptions()
	opts.Limit = 1
	opts.Threshold = 2
	opts.Deduplicate = false

	results, err := searcher.SearchWithOptions(ctx, "what does this repo do", opts)
	if err != nil {
		t.Fatalf("SearchWithOptions failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchWithOptions returned %d results, want 1", len(results))
	}
	if results[0].FilePath != "README.md" {
		t.Fatalf("SearchWithOptions returned %q, want README.md from document-level TQ search", results[0].FilePath)
	}
}

func TestSearcherDocumentLevelFallsBackToLibSQLFileEmbeddings(t *testing.T) {
	t.Setenv("SGREP_VECTOR_BACKEND", "tqmse")

	ctx := context.Background()
	queryEmbedding := searchTestVector(768, 1)
	opened := openSearchTestTQStore(t, ctx, queryEmbedding, false, true)
	defer func() { _ = opened.Close() }()

	searcher := NewWithConfig(Config{
		Store:    opened,
		Embedder: fixedSearchTestEmbedder(t, queryEmbedding),
	})
	opts := DefaultSearchOptions()
	opts.Limit = 1
	opts.Threshold = 2
	opts.Deduplicate = false

	results, err := searcher.SearchWithOptions(ctx, "what does this repo do", opts)
	if err != nil {
		t.Fatalf("SearchWithOptions failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("SearchWithOptions returned %d results, want 1", len(results))
	}
	if results[0].FilePath != "README.md" {
		t.Fatalf("SearchWithOptions returned %q, want README.md from libSQL file-embedding fallback", results[0].FilePath)
	}
}

func openSearchTestTQStore(t *testing.T, ctx context.Context, queryEmbedding []float32, buildFileTQ, storeFileEmbedding bool) store.Storer {
	t.Helper()

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "index.db")
	s, err := store.OpenLibSQL(dbPath)
	if err != nil {
		t.Fatalf("OpenLibSQL failed: %v", err)
	}

	docs := []*store.Document{
		{
			ID:        "README.md:chunk_1",
			FilePath:  "README.md",
			Content:   "sgrep is a semantic code search tool.",
			StartLine: 1,
			EndLine:   4,
			Embedding: searchTestVector(768, 2),
		},
		{
			ID:        "README.md:chunk_2",
			FilePath:  "README.md",
			Content:   "It indexes repositories and answers overview queries.",
			StartLine: 5,
			EndLine:   8,
			Embedding: searchTestVector(768, 3),
		},
		{
			ID:        "internal/other.go:chunk_1",
			FilePath:  "internal/other.go",
			Content:   "package internal\nfunc unrelated() {}",
			StartLine: 1,
			EndLine:   2,
			Embedding: queryEmbedding,
		},
	}
	if err := s.StoreBatch(ctx, docs); err != nil {
		_ = s.Close()
		t.Fatalf("StoreBatch failed: %v", err)
	}

	if storeFileEmbedding {
		if err := s.StoreFileEmbedding(ctx, &store.FileEmbedding{
			FilePath:   "README.md",
			Embedding:  queryEmbedding,
			ChunkCount: 2,
			TotalLines: 8,
		}); err != nil {
			_ = s.Close()
			t.Fatalf("StoreFileEmbedding failed: %v", err)
		}
	}

	if _, err := store.BuildTQVectorStore(ctx, dir,
		[]string{"internal/other.go:chunk_1"},
		[][]float32{queryEmbedding},
		store.TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42},
	); err != nil {
		_ = s.Close()
		t.Fatalf("BuildTQVectorStore failed: %v", err)
	}
	if buildFileTQ {
		if _, err := store.BuildTQFileVectorStore(ctx, dir,
			[]string{"README.md"},
			[][]float32{queryEmbedding},
			store.TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42},
		); err != nil {
			_ = s.Close()
			t.Fatalf("BuildTQFileVectorStore failed: %v", err)
		}
	}

	if err := s.Checkpoint(ctx); err != nil {
		_ = s.Close()
		t.Fatalf("Checkpoint failed: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	opened, err := store.OpenForSearch(dbPath)
	if err != nil {
		t.Fatalf("OpenForSearch failed: %v", err)
	}
	if _, ok := opened.(*store.TQSearchStore); !ok {
		_ = opened.Close()
		t.Fatalf("OpenForSearch returned %T, want *store.TQSearchStore", opened)
	}
	return opened
}

func fixedSearchTestEmbedder(t *testing.T, embedding []float32) *embed.Embedder {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embedding" {
			t.Errorf("unexpected embedding path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding})
	}))
	t.Cleanup(server.Close)

	return embed.NewWithConfig(embed.Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})
}

func searchTestVector(dims int, hotIndex int) []float32 {
	v := make([]float32, dims)
	v[hotIndex%dims] = 1
	return v
}
