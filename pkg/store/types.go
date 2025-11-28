package store

import (
	"context"
	"fmt"
	"os"
)

const defaultDims = 768 // nomic-embed-text dimensions

// inMemoryThreshold is the max vector count for in-memory search.
// Above this, we use native KNN to avoid memory bloat.
const inMemoryThreshold = 50000

// Storer interface for vector stores.
type Storer interface {
	Store(ctx context.Context, doc *Document) error
	StoreBatch(ctx context.Context, docs []*Document) error
	Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error)
	HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error)
	Stats(ctx context.Context) (*Stats, error)
	DeleteByPath(ctx context.Context, filepath string) error
	Close() error
}

// Flusher is an optional interface for stores that buffer writes.
type Flusher interface {
	Flush(ctx context.Context) error
}

// FlushIfNeeded flushes the store if it implements Flusher.
func FlushIfNeeded(ctx context.Context, s Storer) error {
	if f, ok := s.(Flusher); ok {
		return f.Flush(ctx)
	}
	return nil
}

// FTS5Ensurer is an optional interface for stores that support FTS5.
type FTS5Ensurer interface {
	EnsureFTS5() error
}

// EnsureFTS5IfNeeded ensures FTS5 is ready if the store supports it.
func EnsureFTS5IfNeeded(s Storer) error {
	if f, ok := s.(FTS5Ensurer); ok {
		return f.EnsureFTS5()
	}
	return nil
}

// Document represents an indexed code chunk.
type Document struct {
	ID        string
	FilePath  string
	Content   string
	StartLine int
	EndLine   int
	Embedding []float32
	Metadata  map[string]string
	IsTest    bool // Whether this chunk is from a test file
}

// Stats holds index statistics.
type Stats struct {
	Documents int64
	Chunks    int64
	SizeBytes int64
}

// getDims returns the embedding dimensions from env or default.
func getDims() int {
	if v := os.Getenv("SGREP_DIMS"); v != "" {
		var d int
		if _, err := fmt.Sscanf(v, "%d", &d); err == nil && d > 0 {
			return d
		}
	}
	return defaultDims
}
