package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

type denseSearchArtifact interface {
	Search(context.Context, []float32, int, float64) ([]DenseSearchResult, error)
	ScoreByID(context.Context, []float32, []string) (map[string]float64, error)
	Close() error
	Path() string
}

// TQSearchStore wraps a metadata/FTS SQL store with an mmap dense-vector artifact.
// It prefers exact float32 scans for small corpora and TQ-MSE for larger ones.
type TQSearchStore struct {
	base            Storer
	dense           denseSearchArtifact
	fileDense       *TQVectorStore
	loader          DocumentLoader
	fileLoader      FileChunkLoader
	bm25            BM25Scorer
	fileStore       FileEmbeddingStorer
	fileStoreCloser interface{ Close() error }
}

// OpenTQSearchStoreIfAvailable wraps base when a compact vector artifact is present.
func OpenTQSearchStoreIfAvailable(base Storer, repoDir string) (Storer, error) {
	if os.Getenv("SGREP_VECTOR_BACKEND") == "sqlite" || os.Getenv("SGREP_VECTOR_BACKEND") == "libsql" {
		return base, nil
	}
	if !HasTQVectorStore(repoDir) {
		if os.Getenv("SGREP_VECTOR_BACKEND") == "tqmse" {
			_ = base.Close()
			return nil, fmt.Errorf("SGREP_VECTOR_BACKEND=tqmse but %s is missing", TQVectorPath(repoDir))
		}
		return base, nil
	}
	loader, ok := base.(DocumentLoader)
	if !ok {
		if os.Getenv("SGREP_VECTOR_BACKEND") == "tqmse" {
			_ = base.Close()
			return nil, fmt.Errorf("store does not support document hydration for TQ-MSE vectors")
		}
		return base, nil
	}
	var dense denseSearchArtifact
	var err error
	if os.Getenv("SGREP_VECTOR_BACKEND") != "tqmse" && HasMMapVectorStore(repoDir) {
		dense, err = OpenMMapVectorStore(repoDir, getDims())
	}
	if dense == nil || err != nil {
		dense, err = OpenTQVectorStore(repoDir)
	}
	if err != nil {
		if os.Getenv("SGREP_VECTOR_BACKEND") == "tqmse" {
			_ = base.Close()
			return nil, err
		}
		return base, nil
	}
	wrapped := &TQSearchStore{
		base:   base,
		dense:  dense,
		loader: loader,
	}
	if bm25, ok := base.(BM25Scorer); ok {
		wrapped.bm25 = bm25
	}
	if fileLoader, ok := base.(FileChunkLoader); ok {
		wrapped.fileLoader = fileLoader
		if HasTQFileVectorStore(repoDir) {
			if fileDense, err := OpenTQFileVectorStore(repoDir); err == nil {
				wrapped.fileDense = fileDense
			}
		}
	}
	return wrapped, nil
}

// Store delegates writes to the base store. Search wrappers are normally used read-only.
func (s *TQSearchStore) Store(ctx context.Context, doc *Document) error {
	return s.base.Store(ctx, doc)
}

// StoreBatch delegates writes to the base store. Search wrappers are normally used read-only.
func (s *TQSearchStore) StoreBatch(ctx context.Context, docs []*Document) error {
	return s.base.StoreBatch(ctx, docs)
}

// Search runs first-stage dense search against the compact vector artifact.
func (s *TQSearchStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	hits, err := s.dense.Search(ctx, embedding, limit, threshold)
	if err != nil {
		return nil, nil, err
	}
	return s.loadDenseHits(ctx, hits)
}

// HybridSearch combines compressed dense candidates with SQLite FTS scores.
func (s *TQSearchStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error) {
	if queryTerms == "" || s.bm25 == nil {
		if queryTerms == "" {
			return s.Search(ctx, embedding, limit, threshold)
		}
		return s.base.HybridSearch(ctx, embedding, queryTerms, limit, threshold, semanticWeight, bm25Weight)
	}

	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}
	hits, err := s.dense.Search(ctx, embedding, fetchLimit, threshold)
	if err != nil {
		return nil, nil, err
	}

	if searcher, ok := s.bm25.(BM25Searcher); ok {
		lexicalHits, err := searcher.BM25Search(ctx, queryTerms, fetchLimit)
		if err != nil {
			return s.Search(ctx, embedding, limit, threshold)
		}
		if len(hits) == 0 && len(lexicalHits) == 0 {
			return nil, nil, nil
		}
		if len(lexicalHits) > 0 {
			denseIDs := make(map[string]struct{}, len(hits))
			for _, hit := range hits {
				denseIDs[hit.ID] = struct{}{}
			}
			ids := make([]string, 0, len(lexicalHits))
			seen := make(map[string]struct{}, len(lexicalHits))
			for _, hit := range lexicalHits {
				if hit.ID == "" {
					continue
				}
				if _, ok := denseIDs[hit.ID]; ok {
					continue
				}
				if _, ok := seen[hit.ID]; ok {
					continue
				}
				seen[hit.ID] = struct{}{}
				ids = append(ids, hit.ID)
			}
			if len(ids) > 0 {
				if distances, err := s.dense.ScoreByID(ctx, embedding, ids); err == nil {
					for id, distance := range distances {
						hits = append(hits, DenseSearchResult{ID: id, Distance: distance})
					}
				}
			}
		}
		// ScoreByID returns a map. Re-establish the semantic ranking before RRF
		// so lexical-only candidates are not assigned random map iteration ranks.
		sort.Slice(hits, func(i, j int) bool {
			if hits[i].Distance == hits[j].Distance {
				return hits[i].ID < hits[j].ID
			}
			return hits[i].Distance < hits[j].Distance
		})
		return s.loadDenseHits(ctx, fuseHybridCandidates(hits, lexicalHits, limit, semanticWeight, bm25Weight))
	}

	if len(hits) == 0 {
		return nil, nil, nil
	}

	bm25Scores, err := s.bm25.BM25Scores(ctx, queryTerms)
	if err != nil {
		return s.Search(ctx, embedding, limit, threshold)
	}

	for i := range hits {
		hits[i].Distance = semanticWeight*hits[i].Distance + bm25Weight*bm25Scores[hits[i].ID]
	}
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Distance < hits[j].Distance
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return s.loadDenseHits(ctx, hits)
}

// Stats includes the compact vector artifact size when present.
func (s *TQSearchStore) Stats(ctx context.Context) (*Stats, error) {
	stats, err := s.base.Stats(ctx)
	if err != nil {
		return nil, err
	}
	if s.dense != nil {
		if info, err := os.Stat(s.dense.Path()); err == nil {
			stats.SizeBytes += info.Size()
		}
		if s.dense.Path() != TQVectorPath(filepath.Dir(s.dense.Path())) {
			if info, err := os.Stat(TQVectorPath(filepath.Dir(s.dense.Path()))); err == nil {
				stats.SizeBytes += info.Size()
			}
		}
	}
	if s.fileDense != nil {
		if info, err := os.Stat(s.fileDense.path); err == nil {
			stats.SizeBytes += info.Size()
		}
	}
	return stats, nil
}

// DeleteByPath delegates to the base store.
func (s *TQSearchStore) DeleteByPath(ctx context.Context, filepath string) error {
	return s.base.DeleteByPath(ctx, filepath)
}

func (s *TQSearchStore) StoreFileEmbedding(ctx context.Context, fe *FileEmbedding) error {
	if s.fileStore == nil {
		return fmt.Errorf("tq search store does not support file embeddings")
	}
	return s.fileStore.StoreFileEmbedding(ctx, fe)
}

func (s *TQSearchStore) StoreFileEmbeddingBatch(ctx context.Context, fes []*FileEmbedding) error {
	if s.fileStore == nil {
		return fmt.Errorf("tq search store does not support file embeddings")
	}
	return s.fileStore.StoreFileEmbeddingBatch(ctx, fes)
}

func (s *TQSearchStore) SearchFileEmbeddings(ctx context.Context, embedding []float32, limit int, threshold float64) ([]string, []float64, error) {
	if s.fileDense != nil {
		hits, err := s.fileDense.Search(ctx, embedding, limit, threshold)
		if err != nil {
			return nil, nil, err
		}
		paths := make([]string, len(hits))
		distances := make([]float64, len(hits))
		for i, hit := range hits {
			paths[i] = hit.ID
			distances[i] = hit.Distance
		}
		return paths, distances, nil
	}
	if s.fileStore == nil {
		return nil, nil, nil
	}
	return s.fileStore.SearchFileEmbeddings(ctx, embedding, limit, threshold)
}

func (s *TQSearchStore) GetChunksByFilePath(ctx context.Context, filePath string) ([]*Document, error) {
	if s.fileLoader != nil {
		return s.fileLoader.GetChunksByFilePath(ctx, filePath)
	}
	if s.fileStore == nil {
		return nil, nil
	}
	return s.fileStore.GetChunksByFilePath(ctx, filePath)
}

func (s *TQSearchStore) DeleteFileEmbedding(ctx context.Context, filePath string) error {
	if s.fileStore == nil {
		return fmt.Errorf("tq search store does not support file embeddings")
	}
	return s.fileStore.DeleteFileEmbedding(ctx, filePath)
}

// Close closes both the dense artifact and base store.
func (s *TQSearchStore) Close() error {
	var err error
	if s.dense != nil {
		err = s.dense.Close()
	}
	if s.fileDense != nil {
		if closeErr := s.fileDense.Close(); err == nil {
			err = closeErr
		}
	}
	if s.fileStoreCloser != nil {
		if closeErr := s.fileStoreCloser.Close(); err == nil {
			err = closeErr
		}
	}
	if closeErr := s.base.Close(); err == nil {
		err = closeErr
	}
	return err
}

func (s *TQSearchStore) loadDenseHits(ctx context.Context, hits []DenseSearchResult) ([]*Document, []float64, error) {
	if len(hits) == 0 {
		return nil, nil, nil
	}
	ids := make([]string, len(hits))
	for i, hit := range hits {
		ids[i] = hit.ID
	}
	docsByID, err := s.loader.LoadDocumentsByID(ctx, ids)
	if err != nil {
		return nil, nil, err
	}
	docs := make([]*Document, 0, len(hits))
	distances := make([]float64, 0, len(hits))
	for _, hit := range hits {
		doc, ok := docsByID[hit.ID]
		if !ok {
			continue
		}
		docs = append(docs, doc)
		distances = append(distances, hit.Distance)
	}
	return docs, distances, nil
}
