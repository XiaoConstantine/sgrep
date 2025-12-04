package search

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/rerank"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// QueryIntent represents the detected intent behind a search query.
type QueryIntent int

const (
	// IntentConceptual is for general semantic queries (default).
	IntentConceptual QueryIntent = iota
	// IntentDefinition is for queries seeking type/interface definitions.
	IntentDefinition
	// IntentProcedural is for queries seeking implementation/how-to code.
	IntentProcedural
	// IntentExample is for queries explicitly seeking examples.
	IntentExample
)

// String returns the string representation of QueryIntent.
func (qi QueryIntent) String() string {
	switch qi {
	case IntentDefinition:
		return "definition"
	case IntentProcedural:
		return "procedural"
	case IntentExample:
		return "example"
	default:
		return "conceptual"
	}
}

var (
	// Patterns for detecting query intent
	definitionPatterns = regexp.MustCompile(`(?i)\b(type|interface|struct|class|enum|definition|define|what\s+is)\b`)
	proceduralPatterns = regexp.MustCompile(`(?i)\b(how\s+to|implement|handle|process|manage|create|make|build)\b`)
	examplePatterns    = regexp.MustCompile(`(?i)\b(example|sample|demo|usage|show\s+me)\b`)
	// Meta-queries about the repository itself
	metaRepoPatterns = regexp.MustCompile(`(?i)\b(what\s+(does|is)\s+(this|the)\s+(repo|repository|project|codebase)|about\s+(this|the)\s+(repo|project)|purpose\s+of\s+(this|the)|overview|introduction)\b`)
)

// IsMetaRepoQuery returns true if the query is asking about the repository itself.
func IsMetaRepoQuery(query string) bool {
	return metaRepoPatterns.MatchString(query)
}

// DetectQueryIntent analyzes a query string and returns the likely intent.
func DetectQueryIntent(query string) QueryIntent {
	// Check for explicit example requests first
	if examplePatterns.MatchString(query) {
		return IntentExample
	}

	// Check for procedural/how-to queries
	if proceduralPatterns.MatchString(query) {
		return IntentProcedural
	}

	// Check for definition queries
	if definitionPatterns.MatchString(query) {
		return IntentDefinition
	}

	// Default to conceptual for general queries like "error handling"
	// These are often seeking definitions/core implementations
	return IntentConceptual
}

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
	UseRerank      bool    // Enable reranking stage for better precision (default: false)
	RerankTopK     int     // Number of candidates to fetch for reranking (default: 50)
	RerankWeight   float64 // Weight for reranker in RRF fusion (0-1, default: 0.5). Lower = trust vector search more.
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
		UseRerank:      false,
		RerankTopK:     30,  // Fetch 30 candidates for reranking (reduced from 50 for performance)
		RerankWeight:   0.5, // Balance between vector search and reranker (0.5 = equal weight)
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
	Reranker   *rerank.Reranker // Optional reranker for two-stage retrieval
	CacheSize  int
	CacheTTL   time.Duration
	EventBox   *util.EventBox
}

// Searcher handles semantic search with caching.
type Searcher struct {
	store      store.Storer
	embedder   *embed.Embedder
	reranker   *rerank.Reranker // Optional reranker for two-stage retrieval
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
		reranker:   cfg.Reranker,
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

	util.Debugf(util.DebugSummary, "Search: %q (limit=%d, threshold=%.2f, hybrid=%v, rerank=%v)",
		query, opts.Limit, opts.Threshold, opts.UseHybrid, opts.UseRerank)

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

	// Check if this is a meta-repo query (e.g., "what does this repo do")
	// For such queries, try document-level search first
	if IsMetaRepoQuery(query) {
		fileDocsTimer := util.NewTimer("file_level_search")
		fileDocs, fileDists, fileErr := s.searchDocumentLevel(ctx, queryEmb, opts.Limit, opts.Threshold)
		fileDuration := fileDocsTimer.Stop()
		if fileErr == nil && len(fileDocs) > 0 {
			stats.RecordStage("file_level_search", fileDuration, int64(len(fileDocs)))
			util.Debugf(util.DebugSummary, "Meta-query: found %d files via document-level search", len(fileDocs))
			// Use file-level results directly
			docs = fileDocs
			distances = fileDists
		}
	}

	// Fall back to chunk-level search if document-level didn't find results
	if len(docs) == 0 {
		searchTimer := util.NewTimer("vector_search")
		// If reranking is enabled, fetch more candidates
		if opts.UseRerank && s.reranker != nil {
			fetchLimit = opts.RerankTopK
			if fetchLimit < opts.Limit*3 {
				fetchLimit = opts.Limit * 3
			}
		}

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
	}

	// Stage 2: Reranking (if enabled and reranker is available)
	if opts.UseRerank && s.reranker != nil && len(docs) > 0 {
		rerankTimer := util.NewTimer("reranking")
		docs, distances, err = s.rerankResults(ctx, query, docs, distances, opts)
		rerankDuration := rerankTimer.Stop()
		stats.RecordStage("reranking", rerankDuration, int64(len(docs)))
		if err != nil {
			// Log warning but don't fail - fall back to vector results
			fmt.Fprintf(os.Stderr, "reranking failed, using vector results: %v\n", err)
		} else {
			util.Debugf(util.DebugDetailed, "Reranked %d docs in %v", len(docs), rerankDuration.Round(time.Millisecond))
		}
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

// deduplicateResults keeps only the best result per logical file.
// It uses canonical paths to group duplicates from worktrees, vendored code, etc.
func deduplicateResults(results []Result) []Result {
	seen := make(map[string]int) // canonical path -> index in dedupedResults
	var dedupedResults []Result

	for _, r := range results {
		canonical := canonicalPath(r.FilePath)
		if existingIdx, ok := seen[canonical]; ok {
			existing := dedupedResults[existingIdx]
			// Keep the one with better (lower) score, or prefer non-worktree path on tie
			if r.Score < existing.Score ||
				(r.Score == existing.Score && !isWorktreePath(r.FilePath) && isWorktreePath(existing.FilePath)) {
				dedupedResults[existingIdx] = r
			}
		} else {
			seen[canonical] = len(dedupedResults)
			dedupedResults = append(dedupedResults, r)
		}
	}

	return dedupedResults
}

// canonicalPath normalizes a file path by stripping worktree prefixes and other
// duplicate-inducing path components. This allows deduplication of logically
// equivalent files that appear in multiple locations.
//
// Examples:
//   - ".worktrees/feature-branch/pkg/foo.go" -> "pkg/foo.go"
//   - "vendor/github.com/x/y/foo.go" -> "vendor/github.com/x/y/foo.go" (kept as-is)
//   - "pkg/foo.go" -> "pkg/foo.go"
func canonicalPath(path string) string {
	// Handle .worktrees/<branch-name>/... pattern
	// Git worktrees create copies at .worktrees/<name>/<repo-contents>
	if idx := strings.Index(path, ".worktrees/"); idx >= 0 {
		rest := path[idx+len(".worktrees/"):]
		// Find the next slash after the worktree name
		if slashIdx := strings.Index(rest, "/"); slashIdx >= 0 {
			return rest[slashIdx+1:]
		}
	}

	return path
}

// isWorktreePath returns true if the path is inside a .worktrees directory.
func isWorktreePath(path string) bool {
	return strings.Contains(path, ".worktrees/")
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
	_, _ = fmt.Fprintf(h, "%s:%d:%.4f:%v:%v:%.4f:%v:%.4f:%.4f:%v:%d",
		query, opts.Limit, opts.Threshold, opts.IncludeTests, opts.Deduplicate, opts.BoostImpl,
		opts.UseHybrid, opts.SemanticWeight, opts.BM25Weight, opts.UseRerank, opts.RerankTopK)
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

// searchDocumentLevel searches using document-level embeddings and returns chunks from matched files.
// This is used for meta-queries like "what does this repo do" where chunk-level search fails.
func (s *Searcher) searchDocumentLevel(ctx context.Context, queryEmb []float32, limit int, threshold float64) ([]*store.Document, []float64, error) {
	// Check if store supports file embeddings
	feStore, ok := s.store.(store.FileEmbeddingStorer)
	if !ok {
		return nil, nil, nil // Store doesn't support file embeddings
	}

	// Search for relevant files at document level
	filePaths, fileDists, err := feStore.SearchFileEmbeddings(ctx, queryEmb, limit, threshold)
	if err != nil {
		return nil, nil, err
	}

	if len(filePaths) == 0 {
		return nil, nil, nil
	}

	util.Debugf(util.DebugDetailed, "Document-level search found %d files", len(filePaths))

	// For each matched file, get its first chunk (which typically contains the overview/header)
	var docs []*store.Document
	var distances []float64

	for i, fp := range filePaths {
		chunks, err := feStore.GetChunksByFilePath(ctx, fp)
		if err != nil || len(chunks) == 0 {
			continue
		}

		// For meta-queries, prioritize README and root-level files
		// Include first chunk (usually contains overview) and potentially more
		maxChunks := 2
		if strings.HasSuffix(strings.ToLower(fp), "readme.md") {
			maxChunks = 3 // README gets more chunks
		}

		for j := 0; j < maxChunks && j < len(chunks); j++ {
			docs = append(docs, chunks[j])
			distances = append(distances, fileDists[i])
		}

		// Stop if we have enough results
		if len(docs) >= limit*2 {
			break
		}
	}

	return docs, distances, nil
}

// rerankResults uses the reranker to reorder documents using RRF score fusion.
// It combines vector search rankings with cross-encoder scores, applying
// code-specific signal boosting based on query intent.
func (s *Searcher) rerankResults(ctx context.Context, query string, docs []*store.Document, distances []float64, opts SearchOptions) ([]*store.Document, []float64, error) {
	if s.reranker == nil {
		return docs, distances, nil
	}

	// Detect query intent to adjust fusion strategy
	intent := DetectQueryIntent(query)
	util.Debugf(util.DebugDetailed, "Query intent: %s", intent)

	// Performance optimization: limit reranking to top candidates only
	// Reranking is expensive (~70ms per doc with batching), so we limit how many we actually rerank
	maxRerankDocs := 15 // Maximum documents to send to reranker
	if len(docs) > maxRerankDocs {
		// Keep original docs/distances for RRF fusion, but only rerank top N
		util.Debugf(util.DebugDetailed, "Limiting rerank from %d to %d docs for performance", len(docs), maxRerankDocs)
	}

	// Prepare document contents for reranking (only top N)
	rerankCount := len(docs)
	if rerankCount > maxRerankDocs {
		rerankCount = maxRerankDocs
	}
	contents := make([]string, rerankCount)
	for i := 0; i < rerankCount; i++ {
		contents[i] = docs[i].Content
	}

	// Call reranker on subset
	results, err := s.reranker.Rerank(ctx, query, contents)
	if err != nil {
		return docs, distances, err
	}

	// Build a map of reranker scores by document index
	rerankScores := make(map[int]float64)
	for _, r := range results {
		if r.Index >= 0 && r.Index < len(docs) {
			rerankScores[r.Index] = r.Score
		}
	}

	// Scored document with RRF fusion
	type scoredDoc struct {
		doc           *store.Document
		vectorRank    int
		vectorDist    float64
		rerankScore   float64
		codeBoost     float64
		rrfScore      float64
		finalDistance float64
	}

	// Compute RRF scores with code signal boosting
	scored := make([]scoredDoc, len(docs))
	const rrfK = 60.0 // Standard RRF constant to prevent early-rank bias

	// Adjust rerank weight based on query intent
	rerankWeight := opts.RerankWeight
	if rerankWeight == 0 {
		rerankWeight = 0.5 // Default
	}

	// For conceptual/definition queries, trust vector search more
	// For procedural/example queries, trust reranker more
	switch intent {
	case IntentConceptual, IntentDefinition:
		rerankWeight = math.Min(rerankWeight, 0.4) // Cap at 0.4 for definitions
	case IntentExample:
		rerankWeight = math.Max(rerankWeight, 0.6) // At least 0.6 for examples
	}

	for i, doc := range docs {
		vectorRRF := 1.0 / (rrfK + float64(i+1)) // Vector rank (1-indexed)

		// Get reranker score and convert to RRF-compatible value
		rerankScore := rerankScores[i]
		rerankProb := sigmoid(rerankScore)
		// Convert probability to a pseudo-rank (0.99 prob → rank ~1, 0.01 prob → rank ~100)
		rerankPseudoRank := (1.0-rerankProb)*100.0 + 1.0
		rerankRRF := 1.0 / (rrfK + rerankPseudoRank)

		// Compute code signal boost based on file path
		codeBoost := computeCodeSignalBoost(doc.FilePath, intent)

		// RRF fusion: combine vector and reranker with code boost
		// Higher RRF score = better ranking
		rrfScore := ((1.0-rerankWeight)*vectorRRF + rerankWeight*rerankRRF) * codeBoost

		scored[i] = scoredDoc{
			doc:           doc,
			vectorRank:    i + 1,
			vectorDist:    distances[i],
			rerankScore:   rerankScore,
			codeBoost:     codeBoost,
			rrfScore:      rrfScore,
			finalDistance: distances[i], // Preserve original distance for display
		}
	}

	// Sort by RRF score descending (higher = better)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].rrfScore > scored[j].rrfScore
	})

	// Debug: log top results with RRF breakdown
	util.Debugf(util.DebugDetailed, "RRF fusion results (weight=%.2f):", rerankWeight)
	for i := 0; i < len(scored) && i < 10; i++ {
		s := scored[i]
		if s.doc != nil {
			util.Debugf(util.DebugDetailed, "  [%d] RRF=%.4f (vec#%d, rerank=%.2f, boost=%.2f): %s",
				i+1, s.rrfScore, s.vectorRank, s.rerankScore, s.codeBoost, s.doc.FilePath)
		}
	}

	// Limit results
	if len(scored) > opts.Limit {
		scored = scored[:opts.Limit]
	}

	// Reconstruct arrays
	rerankedDocs := make([]*store.Document, len(scored))
	rerankedDists := make([]float64, len(scored))
	for i, s := range scored {
		rerankedDocs[i] = s.doc
		rerankedDists[i] = s.finalDistance
	}

	return rerankedDocs, rerankedDists, nil
}

// computeCodeSignalBoost returns a multiplier based on file path signals.
// Higher boost = better ranking in RRF fusion.
func computeCodeSignalBoost(filePath string, intent QueryIntent) float64 {
	path := strings.ToLower(filePath)
	boost := 1.0

	// Root-level README boost for conceptual queries (highest priority)
	// These are typically project descriptions
	if intent == IntentConceptual {
		if path == "readme.md" || path == "readme" || path == "readme.txt" {
			boost *= 1.5 // +50% for root README
		} else if strings.HasSuffix(path, "/readme.md") && !strings.Contains(path, "example") {
			boost *= 1.2 // +20% for package READMEs
		}
	}

	// Core package boost for definition/conceptual queries
	if intent == IntentConceptual || intent == IntentDefinition {
		// Boost core packages (pkg/*, internal/*, lib/*)
		if strings.HasPrefix(path, "pkg/") || strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "lib/") {
			// Extra boost if the path suggests it's the primary implementation
			// e.g., pkg/errors/errors.go for "error handling" query
			if !strings.Contains(path, "example") && !strings.Contains(path, "test") {
				boost *= 1.3 // +30% for core non-example, non-test code
			}
		}

		// Penalize example directories for definition queries
		if strings.Contains(path, "/example") || strings.HasPrefix(path, "example") {
			boost *= 0.7 // -30% for examples when seeking definitions
		}
	}

	// Example boost for example queries
	if intent == IntentExample {
		if strings.Contains(path, "example") || strings.Contains(path, "demo") {
			boost *= 1.3 // +30% for examples
		}
	}

	// Test file penalty (unless explicitly including tests)
	if strings.HasSuffix(path, "_test.go") || strings.Contains(path, "/test/") {
		boost *= 0.8 // -20% for test files
	}

	// Worktree/duplicate penalty
	if strings.Contains(path, ".worktrees/") || strings.Contains(path, "/.worktrees/") {
		boost *= 0.5 // -50% for worktree duplicates
	}

	// Generated file penalty
	if strings.Contains(path, ".generated.") || strings.Contains(path, ".pb.go") {
		boost *= 0.6 // -40% for generated code
	}

	return boost
}

// SetReranker sets the reranker for two-stage retrieval.
func (s *Searcher) SetReranker(r *rerank.Reranker) {
	s.reranker = r
}

// sigmoid converts a logit to a probability (0-1 range).
// Used to normalize reranker scores which are raw logits (can be negative).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
