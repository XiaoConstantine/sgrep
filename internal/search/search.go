package search

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/XiaoConstantine/sgrep/internal/embed"
	"github.com/XiaoConstantine/sgrep/internal/store"
	"github.com/XiaoConstantine/sgrep/internal/util"
)

// Result represents a search result.
type Result struct {
	FilePath  string  `json:"file"`
	StartLine int     `json:"start"`
	EndLine   int     `json:"end"`
	Score     float64 `json:"score"`
	Content   string  `json:"content,omitempty"`
}

// cachedResult is stored in the query cache.
type cachedResult struct {
	results []Result
	time    time.Time
}

// Config holds searcher configuration (dependency injection pattern).
type Config struct {
	Store      store.Storer
	Embedder   *embed.Embedder
	CacheSize  int
	CacheTTL   time.Duration
	EventBox   *util.EventBox
}

// Searcher handles semantic search with caching.
type Searcher struct {
	store      store.Storer
	embedder   *embed.Embedder
	queryCache *util.QueryCache // L3 cache: query text → results
	eventBox   *util.EventBox
}

// New creates a new searcher with caching enabled.
func New(s store.Storer) *Searcher {
	return NewWithConfig(Config{
		Store:     s,
		CacheSize: 100,
		CacheTTL:  5 * time.Minute,
	})
}

// NewWithOptions creates a searcher with custom cache settings.
// Deprecated: Use NewWithConfig for full control.
func NewWithOptions(s store.Storer, cacheSize int, cacheTTL time.Duration) *Searcher {
	return NewWithConfig(Config{
		Store:     s,
		CacheSize: cacheSize,
		CacheTTL:  cacheTTL,
	})
}

// NewWithConfig creates a searcher with full configuration control.
// This is the preferred constructor for dependency injection.
func NewWithConfig(cfg Config) *Searcher {
	embedder := cfg.Embedder
	if embedder == nil {
		// Create embedder with shared eventBox if provided
		embedCfg := embed.DefaultConfig()
		embedCfg.EventBox = cfg.EventBox
		embedder = embed.NewWithConfig(embedCfg)
	}

	cacheSize := cfg.CacheSize
	if cacheSize == 0 {
		cacheSize = 100
	}

	cacheTTL := cfg.CacheTTL
	if cacheTTL == 0 {
		cacheTTL = 5 * time.Minute
	}

	return &Searcher{
		store:      cfg.Store,
		embedder:   embedder,
		queryCache: util.NewQueryCache(cacheSize, cacheTTL),
		eventBox:   cfg.EventBox,
	}
}

// Search finds code matching the query.
// Uses multi-level caching: L1 (embedding cache in embedder), L3 (query result cache here).
func (s *Searcher) Search(ctx context.Context, query string, limit int, threshold float64) ([]Result, error) {
	// Emit search start event
	if s.eventBox != nil {
		s.eventBox.Set(util.EvtSearchStart, query)
	}

	// Generate cache key from query + params
	cacheKey := s.cacheKey(query, limit, threshold)

	// Check L3 cache (query → results)
	if cached := s.queryCache.Get(cacheKey); cached != nil {
		if cr, ok := cached.(*cachedResult); ok {
			if s.eventBox != nil {
				s.eventBox.Set(util.EvtSearchComplete, len(cr.results))
			}
			return cr.results, nil
		}
	}

	// Generate query embedding (L1 cache in embedder)
	queryEmb, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	// Search store (parallel partitioned search with slabs)
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

	// Cache results (L3)
	s.queryCache.Set(cacheKey, &cachedResult{
		results: results,
		time:    time.Now(),
	})

	// Emit search complete event
	if s.eventBox != nil {
		s.eventBox.Set(util.EvtSearchComplete, len(results))
	}

	return results, nil
}

// cacheKey generates a unique key for query + parameters.
func (s *Searcher) cacheKey(query string, limit int, threshold float64) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s:%d:%.4f", query, limit, threshold)
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// ClearCache clears the query result cache.
func (s *Searcher) ClearCache() {
	s.queryCache.Clear()
}

// CacheStats returns cache statistics.
func (s *Searcher) CacheStats() util.CacheStats {
	return s.queryCache.Stats()
}
