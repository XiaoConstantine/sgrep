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
	UseColBERT     bool    // Enable ColBERT-style late interaction scoring (default: false)
	ColBERTWeight  float64 // Weight for ColBERT score in fusion (0-1, default: 0.3)
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
		UseColBERT:     false,
		ColBERTWeight:  0.3, // ColBERT contribution in fusion (0.3 = 30% ColBERT, 70% other signals)
	}
}

// cachedResult is stored in the query cache.
type cachedResult struct {
	results []Result
	time    time.Time
}

// Config holds searcher configuration (dependency injection pattern).
type Config struct {
	Store         store.Storer
	Embedder      *embed.Embedder
	Reranker      *rerank.Reranker // Optional cross-encoder reranker
	ColBERTScorer *ColBERTScorer   // Optional ColBERT-style late interaction scorer
	CacheSize     int
	CacheTTL      time.Duration
	EventBox      *util.EventBox
}

// Searcher handles semantic search with caching.
type Searcher struct {
	store         store.Storer
	embedder      *embed.Embedder
	reranker      *rerank.Reranker // Optional cross-encoder reranker
	colbertScorer *ColBERTScorer   // Optional ColBERT-style late interaction scorer
	queryCache    *util.QueryCache // L3 cache: query text → results
	eventBox      *util.EventBox
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

	// Initialize ColBERT scorer if not provided but embedder is available
	colbertScorer := cfg.ColBERTScorer
	if colbertScorer == nil && embedder != nil {
		colbertScorer = NewColBERTScorer(embedder)
	}

	// Configure ColBERT scorer with segment store if available
	if colbertScorer != nil {
		if segmentStore, ok := cfg.Store.(store.ColBERTSegmentStorer); ok {
			colbertScorer.SetSegmentStore(segmentStore)
		}
	}

	return &Searcher{
		store:         cfg.Store,
		embedder:      embedder,
		reranker:      cfg.Reranker,
		colbertScorer: colbertScorer,
		queryCache:    util.NewQueryCache(cacheSize, cacheTTL),
		eventBox:      cfg.EventBox,
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

	// Stage 2: Late interaction scoring (ColBERT and/or cross-encoder reranking)
	// CASCADE mode: ColBERT first (preserves keywords), then cross-encoder on top results (semantic).
	// This combination works well even with hybrid search.
	hasReranker := opts.UseRerank && s.reranker != nil
	hasColBERTScorer := opts.UseColBERT && s.colbertScorer != nil
	if (hasReranker || hasColBERTScorer) && len(docs) > 0 {
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

	// Capture trace for slow queries (>500ms) if FlightRecorder is running
	if totalDuration > 500*time.Millisecond {
		if recorder := util.GetFlightRecorder(); recorder.IsStarted() {
			if tracePath, err := recorder.Snapshot("slow-query"); err == nil {
				util.Debugf(util.DebugSummary, "Slow query trace saved: %s (took %v)", tracePath, totalDuration.Round(time.Millisecond))
			}
		}
	}

	return results, nil
}

// deduplicateResults performs smart deduplication:
// 1. Collapses worktree duplicates (same canonical path) - keeps non-worktree version
// 2. Keeps non-overlapping chunks from the same file (different line ranges)
// 3. Collapses overlapping chunks from the same file - keeps best score
func deduplicateResults(results []Result) []Result {
	// Track chunks we've seen for each canonical path
	// Key: canonical path, Value: list of (startLine, endLine, index in dedupedResults)
	fileChunks := make(map[string][]struct {
		start, end int
		idx        int
		isWorktree bool
	})

	var dedupedResults []Result

	for _, r := range results {
		canonical := canonicalPath(r.FilePath)
		isWT := isWorktreePath(r.FilePath)

		chunks, exists := fileChunks[canonical]
		if !exists {
			// First chunk from this file
			fileChunks[canonical] = []struct {
				start, end int
				idx        int
				isWorktree bool
			}{{r.StartLine, r.EndLine, len(dedupedResults), isWT}}
			dedupedResults = append(dedupedResults, r)
			continue
		}

		// Check if this chunk overlaps with any existing chunk from this file
		overlapsIdx := -1
		for _, chunk := range chunks {
			if chunksOverlap(r.StartLine, r.EndLine, chunk.start, chunk.end) {
				overlapsIdx = chunk.idx
				break
			}
		}

		if overlapsIdx >= 0 {
			// Overlapping chunk - keep the better one
			existing := dedupedResults[overlapsIdx]
			existingIsWT := isWorktreePath(existing.FilePath)

			// Prefer: better score, or non-worktree on tie
			if r.Score < existing.Score ||
				(r.Score == existing.Score && !isWT && existingIsWT) {
				dedupedResults[overlapsIdx] = r
				// Update the chunk info
				for i, chunk := range fileChunks[canonical] {
					if chunk.idx == overlapsIdx {
						fileChunks[canonical][i].start = r.StartLine
						fileChunks[canonical][i].end = r.EndLine
						fileChunks[canonical][i].isWorktree = isWT
						break
					}
				}
			}
		} else {
			// Non-overlapping chunk from same file - keep it (more context for agents)
			// But skip if it's a worktree duplicate of a non-worktree chunk
			hasNonWorktreeVersion := false
			if isWT {
				for _, chunk := range chunks {
					if !chunk.isWorktree {
						hasNonWorktreeVersion = true
						break
					}
				}
			}

			if !isWT || !hasNonWorktreeVersion {
				fileChunks[canonical] = append(fileChunks[canonical], struct {
					start, end int
					idx        int
					isWorktree bool
				}{r.StartLine, r.EndLine, len(dedupedResults), isWT})
				dedupedResults = append(dedupedResults, r)
			}
		}
	}

	return dedupedResults
}

// chunksOverlap returns true if two line ranges overlap significantly.
// We consider chunks overlapping if they share more than 50% of lines.
func chunksOverlap(start1, end1, start2, end2 int) bool {
	// No overlap
	if end1 < start2 || end2 < start1 {
		return false
	}

	// Calculate overlap
	overlapStart := start1
	if start2 > overlapStart {
		overlapStart = start2
	}
	overlapEnd := end1
	if end2 < overlapEnd {
		overlapEnd = end2
	}

	overlapLines := overlapEnd - overlapStart + 1
	if overlapLines <= 0 {
		return false
	}

	// Check if overlap is significant (>50% of either chunk)
	chunk1Lines := end1 - start1 + 1
	chunk2Lines := end2 - start2 + 1

	return float64(overlapLines)/float64(chunk1Lines) > 0.5 ||
		float64(overlapLines)/float64(chunk2Lines) > 0.5
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

// rerankResults uses late interaction scoring to reorder documents.
// It supports CASCADE mode when both ColBERT and cross-encoder are enabled:
// 1. ColBERT scores all candidates (fast, preserves keywords)
// 2. Cross-encoder refines top N results (slow, semantic understanding)
// 3. Final fusion combines both signals
func (s *Searcher) rerankResults(ctx context.Context, query string, docs []*store.Document, distances []float64, opts SearchOptions) ([]*store.Document, []float64, error) {
	// Check if we have any reranking capability
	hasReranker := s.reranker != nil && opts.UseRerank
	hasColBERT := s.colbertScorer != nil && opts.UseColBERT

	if !hasReranker && !hasColBERT {
		return docs, distances, nil
	}

	// Detect query intent to adjust fusion strategy
	intent := DetectQueryIntent(query)
	util.Debugf(util.DebugDetailed, "Query intent: %s (reranker=%v, colbert=%v)", intent, hasReranker, hasColBERT)

	// CASCADE MODE: ColBERT first (on more docs), then cross-encoder (on fewer docs)
	// This preserves ColBERT's keyword matching while adding cross-encoder's semantic understanding
	colbertCandidates := 50  // ColBERT can handle more docs (faster)
	crossEncoderCandidates := 20  // Cross-encoder only on top results (slower)

	if !hasColBERT {
		colbertCandidates = 0
	}
	if !hasReranker {
		crossEncoderCandidates = 0
	}

	// Limit to available docs
	rerankCount := len(docs)
	if hasColBERT && rerankCount > colbertCandidates {
		rerankCount = colbertCandidates
	} else if !hasColBERT && rerankCount > crossEncoderCandidates {
		rerankCount = crossEncoderCandidates
	}

	util.Debugf(util.DebugDetailed, "Cascade rerank: %d docs for ColBERT, top %d for cross-encoder", rerankCount, crossEncoderCandidates)

	// Prepare document contents and IDs for ColBERT (or cross-encoder if ColBERT disabled)
	contents := make([]string, rerankCount)
	chunkIDs := make([]string, rerankCount)
	for i := 0; i < rerankCount; i++ {
		contents[i] = docs[i].Content
		chunkIDs[i] = docs[i].ID
	}

	// STAGE 1: ColBERT scoring on all candidates (preserves keyword matches)
	// Uses pre-computed segments when available (fast) or on-demand embedding (slow)
	colbertScores := make(map[int]float64)
	if hasColBERT {
		scores, err := s.colbertScorer.ScoreBatchWithChunkIDs(ctx, query, chunkIDs, contents)
		if err != nil {
			util.Debugf(util.DebugDetailed, "ColBERT error (continuing with other signals): %v", err)
		} else {
			for i, score := range scores {
				colbertScores[i] = score
			}
		}
	}

	// STAGE 2: Cross-encoder on top candidates only (adds semantic understanding)
	// First, sort by ColBERT scores to find top candidates for cross-encoder
	rerankScores := make(map[int]float64)
	if hasReranker && hasColBERT && len(colbertScores) > 0 {
		// Sort indices by ColBERT score (descending)
		type idxScore struct {
			idx   int
			score float64
		}
		colbertRanked := make([]idxScore, 0, len(colbertScores))
		for idx, score := range colbertScores {
			colbertRanked = append(colbertRanked, idxScore{idx, score})
		}
		sort.Slice(colbertRanked, func(i, j int) bool {
			return colbertRanked[i].score > colbertRanked[j].score
		})

		// Take top N for cross-encoder
		topN := crossEncoderCandidates
		if topN > len(colbertRanked) {
			topN = len(colbertRanked)
		}

		topContents := make([]string, topN)
		topIndices := make([]int, topN)
		for i := 0; i < topN; i++ {
			topIndices[i] = colbertRanked[i].idx
			topContents[i] = contents[colbertRanked[i].idx]
		}

		util.Debugf(util.DebugDetailed, "Cross-encoder reranking top %d ColBERT results", topN)

		results, err := s.reranker.Rerank(ctx, query, topContents)
		if err != nil {
			util.Debugf(util.DebugDetailed, "Reranker error (continuing with ColBERT): %v", err)
		} else {
			for _, r := range results {
				if r.Index >= 0 && r.Index < len(topIndices) {
					originalIdx := topIndices[r.Index]
					rerankScores[originalIdx] = r.Score
				}
			}
		}
	} else if hasReranker && !hasColBERT {
		// No ColBERT, just run cross-encoder on all candidates
		results, err := s.reranker.Rerank(ctx, query, contents)
		if err != nil {
			util.Debugf(util.DebugDetailed, "Reranker error: %v", err)
		} else {
			for _, r := range results {
				if r.Index >= 0 && r.Index < len(docs) {
					rerankScores[r.Index] = r.Score
				}
			}
		}
	}

	// Scored document with RRF fusion
	type scoredDoc struct {
		doc           *store.Document
		vectorRank    int
		vectorDist    float64
		rerankScore   float64
		colbertScore  float64
		codeBoost     float64
		rrfScore      float64
		finalDistance float64
	}

	// Compute RRF scores with multi-signal fusion
	scored := make([]scoredDoc, len(docs))
	const rrfK = 60.0 // Standard RRF constant

	// Calculate effective weights based on what's available
	vectorWeight := 0.4
	rerankWeight := opts.RerankWeight
	colbertWeight := opts.ColBERTWeight

	if rerankWeight == 0 {
		rerankWeight = 0.5
	}
	if colbertWeight == 0 {
		colbertWeight = 0.3
	}

	// Adjust weights based on what's enabled
	// IMPORTANT: When hybrid search is enabled, the initial ranking already includes
	// BM25 keyword matching which is highly valuable for code search. We must preserve
	// this signal by giving higher weight to the initial (hybrid) ranking.
	if opts.UseHybrid {
		// Hybrid mode: BM25 keyword matches are highly valuable for code search.
		// The reranker should only make minor adjustments, not override BM25 rankings.
		if hasReranker && hasColBERT {
			// Cross-encoder tends to prefer prose/docs over code, so minimize its weight.
			// ColBERT preserves keyword matching which is critical for code search.
			vectorWeight = 0.70 // Hybrid ranking is primary signal
			rerankWeight = 0.05 // Cross-encoder as tie-breaker only (was 0.15)
			colbertWeight = 0.25 // Preserve ColBERT's keyword matching (was 0.15)
		} else if hasColBERT && !hasReranker {
			vectorWeight = 0.75
			rerankWeight = 0
			colbertWeight = 0.25
		} else if hasReranker && !hasColBERT {
			vectorWeight = 0.75 // BM25 keyword matches are highly valuable
			rerankWeight = 0.25 // Reranker as tie-breaker only
			colbertWeight = 0
		}
	} else {
		// Semantic-only mode: reranker/colbert provide complementary signals
		if hasReranker && hasColBERT {
			vectorWeight = 0.3
			rerankWeight = 0.4
			colbertWeight = 0.3
		} else if hasColBERT && !hasReranker {
			vectorWeight = 0.5
			rerankWeight = 0
			colbertWeight = 0.5
		} else if hasReranker && !hasColBERT {
			vectorWeight = 1.0 - rerankWeight
			colbertWeight = 0
		}
	}

	// Adjust based on query intent
	switch intent {
	case IntentConceptual, IntentDefinition:
		// Trust vector search more for definitions
		vectorWeight = math.Min(vectorWeight+0.1, 0.6)
		if hasReranker {
			rerankWeight = math.Max(rerankWeight-0.05, 0.2)
		} else if hasColBERT {
			// When no reranker, boost ColBERT for conceptual queries
			colbertWeight = math.Min(colbertWeight+0.2, 0.5)
		}
	case IntentExample:
		// Trust late interaction more for examples
		if hasColBERT {
			colbertWeight = math.Min(colbertWeight+0.1, 0.5)
		}
		if hasReranker {
			rerankWeight = math.Min(rerankWeight+0.1, 0.5)
		}
	}

	// Normalize weights
	totalWeight := vectorWeight + rerankWeight + colbertWeight
	vectorWeight /= totalWeight
	rerankWeight /= totalWeight
	colbertWeight /= totalWeight

	util.Debugf(util.DebugDetailed, "Fusion weights: vector=%.2f, reranker=%.2f, colbert=%.2f", vectorWeight, rerankWeight, colbertWeight)

	for i, doc := range docs {
		vectorRRF := 1.0 / (rrfK + float64(i+1))

		// Cross-encoder contribution
		var rerankRRF float64
		if rerankScore, ok := rerankScores[i]; ok {
			rerankProb := sigmoid(rerankScore)
			rerankPseudoRank := (1.0-rerankProb)*100.0 + 1.0
			rerankRRF = 1.0 / (rrfK + rerankPseudoRank)
		}

		// ColBERT contribution (score is already 0-1 range)
		var colbertRRF float64
		if colbertScore, ok := colbertScores[i]; ok {
			// Convert ColBERT score (higher=better, ~0.5-1.0 range) to pseudo-rank
			colbertPseudoRank := (1.0-colbertScore)*50.0 + 1.0
			colbertRRF = 1.0 / (rrfK + colbertPseudoRank)
		}

		// Code signal boost
		codeBoost := computeCodeSignalBoost(doc.FilePath, intent)

		// Multi-signal RRF fusion
		rrfScore := (vectorWeight*vectorRRF + rerankWeight*rerankRRF + colbertWeight*colbertRRF) * codeBoost

		scored[i] = scoredDoc{
			doc:           doc,
			vectorRank:    i + 1,
			vectorDist:    distances[i],
			rerankScore:   rerankScores[i],
			colbertScore:  colbertScores[i],
			codeBoost:     codeBoost,
			rrfScore:      rrfScore,
			finalDistance: distances[i],
		}
	}

	// Sort by RRF score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].rrfScore > scored[j].rrfScore
	})

	// Debug logging
	util.Debugf(util.DebugDetailed, "RRF fusion results:")
	for i := 0; i < len(scored) && i < 10; i++ {
		s := scored[i]
		if s.doc != nil {
			util.Debugf(util.DebugDetailed, "  [%d] RRF=%.4f (vec#%d, rerank=%.2f, colbert=%.2f, boost=%.2f): %s",
				i+1, s.rrfScore, s.vectorRank, s.rerankScore, s.colbertScore, s.codeBoost, s.doc.FilePath)
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
