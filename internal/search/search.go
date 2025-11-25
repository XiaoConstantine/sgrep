package search

import (
	"context"

	"github.com/XiaoConstantine/sgrep/internal/embed"
	"github.com/XiaoConstantine/sgrep/internal/store"
)

// Result represents a search result.
type Result struct {
	FilePath  string  `json:"file"`
	StartLine int     `json:"start"`
	EndLine   int     `json:"end"`
	Score     float64 `json:"score"`
	Content   string  `json:"content,omitempty"`
}

// Searcher handles semantic search.
type Searcher struct {
	store    *store.Store
	embedder *embed.Embedder
}

// New creates a new searcher.
func New(s *store.Store) *Searcher {
	return &Searcher{
		store:    s,
		embedder: embed.New(),
	}
}

// Search finds code matching the query.
func (s *Searcher) Search(ctx context.Context, query string, limit int, threshold float64) ([]Result, error) {
	// Generate query embedding
	queryEmb, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	// Search store
	docs, distances, err := s.store.Search(ctx, queryEmb, limit, threshold)
	if err != nil {
		return nil, err
	}

	// Convert to results
	results := make([]Result, len(docs))
	for i, doc := range docs {
		results[i] = Result{
			FilePath:  doc.FilePath,
			StartLine: doc.StartLine,
			EndLine:   doc.EndLine,
			Score:     distances[i],
			Content:   doc.Content,
		}
	}

	return results, nil
}
