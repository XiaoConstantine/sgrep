package search

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// Result represents a search result.
type Result struct {
	FilePath  string  `json:"file"`
	StartLine int     `json:"start"`
	EndLine   int     `json:"end"`
	Score     float64 `json:"score"`
	Content   string  `json:"content,omitempty"`
	IsTest    bool    `json:"is_test,omitempty"`
}

// SearchOptions configures search behavior.
type SearchOptions struct {
	Limit          int
	Threshold      float64
	IncludeTests   bool    // Include test files in results (default: false)
	Deduplicate    bool    // Deduplicate results by file (default: true)
	BoostImpl      float64 // Boost factor for implementation files (default: 0.85, lower = better ranking)
	UseHybrid      bool    // Enable hybrid search combining semantic + BM25 (default: false)
	SemanticWeight float64 // Weight for semantic score in hybrid mode (default: 0.6)
	BM25Weight     float64 // Weight for BM25 score in hybrid mode (default: 0.4)
}

// DefaultSearchOptions returns sensible default options.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		Limit:          10,
		Threshold:      0.65, // Cosine distance threshold (0 = identical, 2 = opposite)
		IncludeTests:   false,
		Deduplicate:    true,
		BoostImpl:      0.92, // Implementation files get 8% score boost (lower distance = better)
		UseHybrid:      false,
		SemanticWeight: 0.6,
		BM25Weight:     0.4,
	}
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

// Search finds code matching the query (backward compatible).
// Uses multi-level caching: L1 (embedding cache in embedder), L3 (query result cache here).
func (s *Searcher) Search(ctx context.Context, query string, limit int, threshold float64) ([]Result, error) {
	opts := DefaultSearchOptions()
	opts.Limit = limit
	opts.Threshold = threshold
	return s.SearchWithOptions(ctx, query, opts)
}

// SearchWithOptions finds code matching the query with full control over search behavior.
func (s *Searcher) SearchWithOptions(ctx context.Context, query string, opts SearchOptions) ([]Result, error) {
	debugLevel := util.GetDebugLevel()
	stats := util.NewTimingStats(debugLevel)
	totalTimer := util.NewTimer("total_search")

	// Emit search start event
	if s.eventBox != nil {
		s.eventBox.Set(util.EvtSearchStart, query)
	}

	util.Debugf(util.DebugSummary, "Search: %q (limit=%d, threshold=%.2f, hybrid=%v)",
		query, opts.Limit, opts.Threshold, opts.UseHybrid)

	// Generate cache key from query + all options
	cacheKey := s.cacheKeyWithOpts(query, opts)

	// Check L3 cache (query → results)
	if cached := s.queryCache.Get(cacheKey); cached != nil {
		if cr, ok := cached.(*cachedResult); ok {
			util.Debugf(util.DebugSummary, "Cache hit: returning %d cached results", len(cr.results))
			if s.eventBox != nil {
				s.eventBox.Set(util.EvtSearchComplete, len(cr.results))
			}
			return cr.results, nil
		}
	}

	// Generate query embedding (L1 cache in embedder)
	embedTimer := util.NewTimer("query_embedding")
	queryEmb, err := s.embedder.Embed(ctx, query)
	embedDuration := embedTimer.Stop()
	stats.RecordStage("query_embedding", embedDuration, 1)
	if err != nil {
		return nil, err
	}

	var docs []*store.Document
	var distances []float64

	// Request more results than limit to allow for filtering
	fetchLimit := opts.Limit * 3
	if !opts.IncludeTests {
		fetchLimit = opts.Limit * 5 // Need more results when filtering tests
	}

	searchTimer := util.NewTimer("vector_search")
	if opts.UseHybrid {
		// Hybrid search: combine semantic + BM25
		queryTerms := ExtractSearchTerms(query)
		docs, distances, err = s.store.HybridSearch(ctx, queryEmb, queryTerms,
			fetchLimit, opts.Threshold, opts.SemanticWeight, opts.BM25Weight)
	} else {
		// Semantic-only search
		docs, distances, err = s.store.Search(ctx, queryEmb, fetchLimit, opts.Threshold)
	}
	searchDuration := searchTimer.Stop()
	stats.RecordStage("vector_search", searchDuration, int64(len(docs)))
	if err != nil {
		return nil, err
	}

	// Convert to results with IsTest flag
	filterTimer := util.NewTimer("filtering")
	var results []Result
	for i, doc := range docs {
		// Filter out test files if not requested
		if !opts.IncludeTests && doc.IsTest {
			continue
		}

		score := distances[i]
		// Apply boost to implementation files (lower score = better)
		// Skip boost in hybrid mode since BM25 already has its own ranking
		if !opts.UseHybrid && !doc.IsTest && opts.BoostImpl > 0 && opts.BoostImpl < 1.0 {
			score *= opts.BoostImpl
		}

		results = append(results, Result{
			FilePath:  doc.FilePath,
			StartLine: doc.StartLine,
			EndLine:   doc.EndLine,
			Score:     score,
			Content:   doc.Content,
			IsTest:    doc.IsTest,
		})
	}

	// Re-sort by adjusted score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	// Deduplicate by file if requested
	if opts.Deduplicate {
		results = deduplicateResults(results)
	}

	// Apply limit after filtering and deduplication
	if len(results) > opts.Limit {
		results = results[:opts.Limit]
	}
	filterDuration := filterTimer.Stop()
	stats.RecordStage("filtering", filterDuration, int64(len(results)))

	// Cache results (L3)
	s.queryCache.Set(cacheKey, &cachedResult{
		results: results,
		time:    time.Now(),
	})

	// Emit search complete event
	if s.eventBox != nil {
		s.eventBox.Set(util.EvtSearchComplete, len(results))
	}

	// Print timing summary
	totalDuration := totalTimer.Stop()
	stats.RecordStage("total", totalDuration, int64(len(results)))
	if debugLevel >= util.DebugSummary {
		stats.PrintSummary()
		util.Debugf(util.DebugSummary, "Search completed: %d results in %v", len(results), totalDuration.Round(time.Millisecond))
	}

	return results, nil
}

// deduplicateResults keeps only the best result per file.
func deduplicateResults(results []Result) []Result {
	seen := make(map[string]int) // filepath -> index in dedupedResults
	var dedupedResults []Result

	for _, r := range results {
		if existingIdx, ok := seen[r.FilePath]; ok {
			// Keep the one with better (lower) score
			if r.Score < dedupedResults[existingIdx].Score {
				dedupedResults[existingIdx] = r
			}
		} else {
			seen[r.FilePath] = len(dedupedResults)
			dedupedResults = append(dedupedResults, r)
		}
	}

	return dedupedResults
}

// cacheKey generates a unique key for query + parameters.
func (s *Searcher) cacheKey(query string, limit int, threshold float64) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s:%d:%.4f", query, limit, threshold)
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// cacheKeyWithOpts generates a unique key for query + all options.
func (s *Searcher) cacheKeyWithOpts(query string, opts SearchOptions) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s:%d:%.4f:%v:%v:%.4f:%v:%.4f:%.4f",
		query, opts.Limit, opts.Threshold, opts.IncludeTests, opts.Deduplicate, opts.BoostImpl,
		opts.UseHybrid, opts.SemanticWeight, opts.BM25Weight)
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
