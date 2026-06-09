package store

import (
	"context"
	"fmt"
	"os"
	"sort"
)

// TQSearchStore wraps a metadata/FTS SQL store with a compact dense-vector store.
type TQSearchStore struct {
	base   Storer
	dense  *TQVectorStore
	loader DocumentLoader
	bm25   BM25Scorer
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
	dense, err := OpenTQVectorStore(repoDir)
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
	if len(hits) == 0 {
		return nil, nil, nil
	}

	bm25Scores, err := s.bm25.BM25Scores(ctx, queryTerms)
	if err != nil {
		return nil, nil, err
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
		if info, err := os.Stat(s.dense.path); err == nil {
			stats.SizeBytes += info.Size()
		}
	}
	return stats, nil
}

// DeleteByPath delegates to the base store.
func (s *TQSearchStore) DeleteByPath(ctx context.Context, filepath string) error {
	return s.base.DeleteByPath(ctx, filepath)
}

// Close closes both the dense artifact and base store.
func (s *TQSearchStore) Close() error {
	var err error
	if s.dense != nil {
		err = s.dense.Close()
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
