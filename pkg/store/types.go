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

// Checkpointer is an optional interface for stores that can checkpoint WAL data.
type Checkpointer interface {
	Checkpoint(ctx context.Context) error
}

// CheckpointIfNeeded checkpoints the store if supported.
func CheckpointIfNeeded(ctx context.Context, s Storer) error {
	if c, ok := s.(Checkpointer); ok {
		return c.Checkpoint(ctx)
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

// FileEmbeddingStorer is an optional interface for stores that support document-level embeddings.
type FileEmbeddingStorer interface {
	StoreFileEmbedding(ctx context.Context, fe *FileEmbedding) error
	StoreFileEmbeddingBatch(ctx context.Context, fes []*FileEmbedding) error
	SearchFileEmbeddings(ctx context.Context, embedding []float32, limit int, threshold float64) ([]string, []float64, error)
	GetChunksByFilePath(ctx context.Context, filePath string) ([]*Document, error)
	DeleteFileEmbedding(ctx context.Context, filePath string) error
}

// FileEmbeddingComputer is an optional interface for stores that can compute file embeddings from chunks.
type FileEmbeddingComputer interface {
	ComputeAndStoreFileEmbeddings(ctx context.Context) (int, error)
}

// VectorExporter is an optional interface for stores that can export all vectors for MMap storage.
type VectorExporter interface {
	// ExportAllVectors returns all chunk IDs and their corresponding embeddings.
	// This is used to export vectors to MMap format for zero-copy access.
	ExportAllVectors(ctx context.Context) (chunkIDs []string, embeddings [][]float32, err error)
}

// DenseSearchResult is a vector-search hit before document hydration.
type DenseSearchResult struct {
	ID       string
	Distance float64
}

// DenseVectorSearcher is an optional external vector artifact used for first-stage search.
type DenseVectorSearcher interface {
	Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]DenseSearchResult, error)
	Close() error
}

// DocumentLoader hydrates chunks by ID while preserving metadata/content in the SQL store.
type DocumentLoader interface {
	LoadDocumentsByID(ctx context.Context, ids []string) (map[string]*Document, error)
}

// BM25Scorer exposes FTS scores without tying dense candidate generation to SQLite vectors.
type BM25Scorer interface {
	BM25Scores(ctx context.Context, queryTerms string) (map[string]float64, error)
}

// ChunkInfo contains minimal chunk data needed for ColBERT segment computation.
type ChunkInfo struct {
	ID          string
	Content     string
	Description string
}

// ColBERTSegmentStorer is an optional interface for stores that support pre-computed ColBERT segments.
// Pre-computing segment embeddings during indexing enables fast MaxSim scoring at query time.
type ColBERTSegmentStorer interface {
	// StoreColBERTSegments stores pre-computed segment embeddings for a chunk.
	// Each chunk can have multiple segments (lines, functions, etc.) with their own embeddings.
	StoreColBERTSegments(ctx context.Context, chunkID string, segments []ColBERTSegment) error

	// StoreColBERTSegmentsBatch stores segments for multiple chunks efficiently.
	StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]ColBERTSegment) error

	// GetColBERTSegments retrieves pre-computed segment embeddings for a chunk.
	GetColBERTSegments(ctx context.Context, chunkID string) ([]ColBERTSegment, error)

	// GetColBERTSegmentsBatch retrieves segments for multiple chunks efficiently.
	GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]ColBERTSegment, error)

	// DeleteColBERTSegments removes segment embeddings for a chunk.
	DeleteColBERTSegments(ctx context.Context, chunkID string) error

	// HasColBERTSegments checks if ColBERT segments exist for any chunks.
	HasColBERTSegments(ctx context.Context) (bool, error)

	// GetChunksForColBERT retrieves chunks in paginated batches for ColBERT preindexing.
	// This avoids memory issues with large repos by not loading all chunks at once.
	// Returns chunks starting at offset, up to batchSize. Returns empty slice when done.
	GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]ChunkInfo, error)
}

// ColBERTSegment represents a pre-computed segment embedding for ColBERT scoring.
// Supports float32 staging plus compact persisted codecs for query-time scoring.
type ColBERTSegment struct {
	SegmentIdx int       // Index within the chunk (0, 1, 2, ...)
	Text       string    // Original segment text (for debugging)
	Embedding  []float32 // Pre-normalized embedding vector (float32, 3072 bytes for 768 dims)

	// Quantized storage (int8, 768 bytes for 768 dims = 4x compression)
	// If EmbeddingInt8 is set, Embedding may be nil to save memory during retrieval.
	EmbeddingInt8 []int8  // Quantized embedding values
	QuantScale    float32 // Scale factor for dequantization
	QuantMin      float32 // Minimum value for dequantization

	// Product quantized storage (for example m=6 => 6 bytes/vector).
	// Query-time scoring uses ADC tables via pkg/util/pq.go.
	PQCodes []byte

	// TQ-MSE storage uses bit-packed per-coordinate scalar codes after a
	// deterministic orthogonal rotation.
	TQCodes []byte
}

// FileEmbedding represents a document-level embedding for a file.
type FileEmbedding struct {
	FilePath   string
	Embedding  []float32
	ChunkCount int
	TotalLines int
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
