package search

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// ColBERTScorer implements late interaction scoring similar to ColBERT.
// Instead of single-vector similarity, it computes MaxSim between query
// terms and document segments for more precise relevance scoring.
//
// Supports two modes:
// 1. Pre-computed segments: Load from store (fast, ~1-5ms per query)
// 2. On-demand embedding: Generate at query time (slow, ~100ms per doc)
type ColBERTScorer struct {
	embedder         *embed.Embedder
	segmentStore     store.ColBERTSegmentStorer // Optional: for pre-computed segments
	cache            *segmentCache
	adaptiveSegments bool
	pq               *util.ProductQuantizer
	tq               *util.TQMSEQuantizer
	pqExactRescoreK  int
}

const adaptiveExactRescorePoolMinSim = 0.90

type preparedQueryTerm struct {
	embedding []float32
	sum       float64
	pqTable   [][]float64
	tqQuery   util.TQMSEQuery
}

// segmentCache caches PRE-NORMALIZED segment embeddings to avoid recomputation.
// Storing normalized vectors enables fast similarity via dot product.
type segmentCache struct {
	mu    sync.RWMutex
	items map[string][]float32 // segment text -> NORMALIZED embedding
	max   int
}

func newSegmentCache(maxSize int) *segmentCache {
	return &segmentCache{
		items: make(map[string][]float32),
		max:   maxSize,
	}
}

func (c *segmentCache) get(key string) []float32 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.items[key]
}

func (c *segmentCache) set(key string, emb []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.items) >= c.max {
		// Simple eviction: clear half
		count := 0
		for k := range c.items {
			delete(c.items, k)
			count++
			if count >= c.max/2 {
				break
			}
		}
	}
	// Store normalized copy for fast dot-product similarity
	c.items[key] = util.NormalizeVectorCopy(emb)
}

// NewColBERTScorer creates a new ColBERT-style scorer.
func NewColBERTScorer(embedder *embed.Embedder) *ColBERTScorer {
	return &ColBERTScorer{
		embedder: embedder,
		cache:    newSegmentCache(1000),
	}
}

// SetSegmentStore sets the store for pre-computed segments.
// When set, the scorer will use pre-computed embeddings instead of generating on-demand.
func (c *ColBERTScorer) SetSegmentStore(s store.ColBERTSegmentStorer) {
	c.segmentStore = s
}

// SetAdaptiveSegments enables token-aware sqrt(M) segment budgets.
func (c *ColBERTScorer) SetAdaptiveSegments(enabled bool) {
	c.adaptiveSegments = enabled
}

// SetProductQuantizer configures PQ ADC scoring for segments that carry PQ codes.
func (c *ColBERTScorer) SetProductQuantizer(pq *util.ProductQuantizer) {
	c.pq = pq
}

// SetTQMSEQuantizer configures TQ-MSE scoring for segments that carry TQ codes.
func (c *ColBERTScorer) SetTQMSEQuantizer(tq *util.TQMSEQuantizer) {
	c.tq = tq
}

// SetPQExactRescoreTopK configures exact rescoring for the top PQ-ranked docs.
// Set to 0 to disable the second pass.
func (c *ColBERTScorer) SetPQExactRescoreTopK(k int) {
	if k < 0 {
		k = 0
	}
	c.pqExactRescoreK = k
}

// Score computes ColBERT-style MaxSim score between query and document.
// Returns a score where higher = more relevant.
func (c *ColBERTScorer) Score(ctx context.Context, query string, docContent string) (float64, error) {
	// Decompose query into terms
	queryTerms := decomposeQuery(query)
	if len(queryTerms) == 0 {
		return 0, nil
	}

	// Decompose document into segments
	docSegments := c.documentSegments(docContent)
	if len(docSegments) == 0 {
		return 0, nil
	}

	// Get embeddings for query terms
	queryEmbeddings, err := c.embedTexts(ctx, queryTerms, true)
	if err != nil {
		return 0, err
	}

	// Get embeddings for document segments
	docEmbeddings, err := c.embedTexts(ctx, docSegments, false)
	if err != nil {
		return 0, err
	}

	// Compute MaxSim: for each query term, find max similarity to any doc segment
	// Uses dot product for pre-normalized vectors (4x faster than cosine)
	var totalScore float64
	for _, qEmb := range queryEmbeddings {
		maxSim := float64(-1)
		for _, dEmb := range docEmbeddings {
			sim := dotProductSimilarity(qEmb, dEmb)
			if sim > maxSim {
				maxSim = sim
			}
		}
		if maxSim > 0 {
			totalScore += maxSim
		}
	}

	// Normalize by number of query terms
	return totalScore / float64(len(queryTerms)), nil
}

// ScoreBatch scores multiple documents against a query efficiently.
// Uses parallel scoring with semaphore-controlled concurrency.
// Returns scores in the same order as documents.
func (c *ColBERTScorer) ScoreBatch(ctx context.Context, query string, documents []string) ([]float64, error) {
	// Decompose query into terms
	queryTerms := decomposeQuery(query)
	if len(queryTerms) == 0 {
		return make([]float64, len(documents)), nil
	}

	// Get query term embeddings (computed once, normalized via cache)
	queryEmbeddings, err := c.embedTexts(ctx, queryTerms, true)
	if err != nil {
		return nil, err
	}

	util.Debugf(util.DebugDetailed, "ColBERT: %d query terms, scoring %d docs (parallel)", len(queryTerms), len(documents))

	// Parallel scoring with semaphore
	scores := make([]float64, len(documents))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	// Semaphore: 8 concurrent scorers (matches embed.go pattern)
	sem := make(chan struct{}, 8)

	for i, doc := range documents {
		i, doc := i, doc // Capture loop variables
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()

			// Check for cancellation
			select {
			case <-ctx.Done():
				mu.Lock()
				if firstErr == nil {
					firstErr = ctx.Err()
				}
				mu.Unlock()
				return
			default:
			}

			docSegments := c.documentSegments(doc)
			if len(docSegments) == 0 {
				return
			}

			docEmbeddings, err := c.embedTexts(ctx, docSegments, false)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}

			// MaxSim computation with pre-allocated buffer
			distances := make([]float64, len(docEmbeddings))
			var totalScore float64
			for _, qEmb := range queryEmbeddings {
				maxSim := maxSimBatch(qEmb, docEmbeddings, distances)
				if maxSim > 0 {
					totalScore += maxSim
				}
			}
			scores[i] = totalScore / float64(len(queryTerms))
		})
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return scores, nil
}

// ScoreBatchWithChunkIDs scores documents using pre-computed segment embeddings.
// This is the FAST path (~1-5ms total vs ~100ms per doc with on-demand embedding).
// Falls back to ScoreBatch if pre-computed segments aren't available.
func (c *ColBERTScorer) ScoreBatchWithChunkIDs(ctx context.Context, query string, chunkIDs []string, documents []string) ([]float64, error) {
	// If no segment store, fall back to on-demand embedding
	if c.segmentStore == nil {
		util.Debugf(util.DebugDetailed, "ColBERT: no segment store, using on-demand embedding")
		return c.ScoreBatch(ctx, query, documents)
	}

	// Decompose query into terms
	queryTerms := decomposeQuery(query)
	if len(queryTerms) == 0 {
		return make([]float64, len(documents)), nil
	}

	// Get query term embeddings (computed once, normalized via cache)
	queryEmbeddings, err := c.embedTexts(ctx, queryTerms, true)
	if err != nil {
		return nil, err
	}
	preparedTerms := prepareQueryTermsWithCodecs(queryEmbeddings, c.pq, c.tq)

	// Batch load pre-computed segment embeddings for all chunks
	segmentMap, err := c.segmentStore.GetColBERTSegmentsBatch(ctx, chunkIDs)
	if err != nil {
		util.Debugf(util.DebugDetailed, "ColBERT: failed to load segments, falling back: %v", err)
		return c.ScoreBatch(ctx, query, documents)
	}

	// Check if we have pre-computed segments
	hasPrecomputed := 0
	for _, segs := range segmentMap {
		if len(segs) > 0 {
			hasPrecomputed++
		}
	}

	if hasPrecomputed == 0 {
		util.Debugf(util.DebugDetailed, "ColBERT: no pre-computed segments found, using on-demand embedding")
		return c.ScoreBatch(ctx, query, documents)
	}

	util.Debugf(util.DebugDetailed, "ColBERT: using %d/%d pre-computed segment sets", hasPrecomputed, len(chunkIDs))

	// Score documents using pre-computed segments (FAST: pure CPU, no network)
	scores := make([]float64, len(documents))

	for i, chunkID := range chunkIDs {
		segments := segmentMap[chunkID]
		if len(segments) == 0 {
			// Fall back to on-demand for this document
			docSegments := c.documentSegments(documents[i])
			if len(docSegments) == 0 {
				continue
			}
			docEmbeddings, err := c.embedTexts(ctx, docSegments, false)
			if err != nil {
				continue
			}
			segments = make([]store.ColBERTSegment, len(docEmbeddings))
			for j, emb := range docEmbeddings {
				segments[j] = store.ColBERTSegment{Embedding: emb}
			}
		}

		// Compute MaxSim score using int8 quantized embeddings if available
		var totalScore float64
		for _, term := range preparedTerms {
			maxSim := maxSimPrepared(term, segments, c.pq, c.tq)
			if maxSim > 0 {
				totalScore += maxSim
			}
		}
		scores[i] = totalScore / float64(len(queryTerms))
	}

	if err := c.exactRescoreTopPQDocs(ctx, queryEmbeddings, chunkIDs, documents, segmentMap, scores); err != nil {
		util.Debugf(util.DebugDetailed, "ColBERT: skipping PQ exact rescore: %v", err)
	}

	return scores, nil
}

func (c *ColBERTScorer) exactRescoreTopPQDocs(ctx context.Context, queryEmbeddings [][]float32, chunkIDs, documents []string, segmentMap map[string][]store.ColBERTSegment, scores []float64) error {
	if c == nil || c.embedder == nil || c.pq == nil || c.pqExactRescoreK <= 0 {
		return nil
	}
	totalStart := time.Now()

	type candidate struct {
		idx   int
		score float64
	}

	candidates := make([]candidate, 0, len(chunkIDs))
	for i, chunkID := range chunkIDs {
		if i >= len(documents) || i >= len(scores) {
			break
		}
		if !hasPQSegments(segmentMap[chunkID]) {
			continue
		}
		candidates = append(candidates, candidate{idx: i, score: scores[i]})
	}
	if len(candidates) == 0 {
		return nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	if len(candidates) > c.pqExactRescoreK {
		candidates = candidates[:c.pqExactRescoreK]
	}

	type docRange struct {
		idx   int
		start int
		end   int
	}

	ranges := make([]docRange, 0, len(candidates))
	allSegments := make([]string, 0, len(candidates)*4)
	decomposeStart := time.Now()
	for _, cand := range candidates {
		docSegments := c.exactRescoreDocumentSegments(documents[cand.idx])
		if len(docSegments) == 0 {
			continue
		}
		start := len(allSegments)
		allSegments = append(allSegments, docSegments...)
		ranges = append(ranges, docRange{
			idx:   cand.idx,
			start: start,
			end:   len(allSegments),
		})
	}
	decomposeDuration := time.Since(decomposeStart)
	if len(allSegments) == 0 {
		return nil
	}

	cacheHits, cacheMisses := c.segmentCacheStats(allSegments)
	embedStart := time.Now()
	docEmbeddings, err := c.embedTexts(ctx, allSegments, false)
	embedDuration := time.Since(embedStart)
	if err != nil {
		return err
	}

	scoreStart := time.Now()
	pooledSegments := 0
	for _, r := range ranges {
		docSegmentTexts := allSegments[r.start:r.end]
		docSegmentEmbeddings := docEmbeddings[r.start:r.end]
		scoreEmbeddings := c.exactRescoreEmbeddings(docSegmentTexts, docSegmentEmbeddings)
		pooledSegments += len(scoreEmbeddings)
		scores[r.idx] = exactColBERTScore(queryEmbeddings, scoreEmbeddings)
	}
	scoreDuration := time.Since(scoreStart)
	util.Debugf(util.DebugSummary,
		"PQ exact rescore: docs=%d raw_segments=%d pooled_segments=%d cache_hits=%d cache_misses=%d decompose=%v embed=%v score=%v total=%v",
		len(ranges),
		len(allSegments),
		pooledSegments,
		cacheHits,
		cacheMisses,
		decomposeDuration.Round(time.Millisecond),
		embedDuration.Round(time.Millisecond),
		scoreDuration.Round(time.Millisecond),
		time.Since(totalStart).Round(time.Millisecond),
	)
	return nil
}

func (c *ColBERTScorer) exactRescoreDocumentSegments(content string) []string {
	if c != nil && c.adaptiveSegments {
		return DecomposeDocumentRaw(content)
	}
	return c.documentSegments(content)
}

func (c *ColBERTScorer) exactRescoreEmbeddings(segmentTexts []string, embeddings [][]float32) [][]float32 {
	if c == nil || !c.adaptiveSegments {
		return embeddings
	}
	target := AdaptiveSegmentBudgetFromRawCount(len(segmentTexts))
	if len(embeddings) <= target {
		return embeddings
	}

	segments := make([]store.ColBERTSegment, 0, len(segmentTexts))
	for i := range segmentTexts {
		if i >= len(embeddings) || len(embeddings[i]) == 0 {
			continue
		}
		segments = append(segments, store.ColBERTSegment{
			SegmentIdx: i,
			Text:       segmentTexts[i],
			Embedding:  embeddings[i],
		})
	}
	if len(segments) <= target {
		return embeddings
	}

	pooled := store.NewSegmentPooler(target, adaptiveExactRescorePoolMinSim).PoolAndMerge(segments)
	pooledEmbeddings := make([][]float32, 0, len(pooled))
	for _, seg := range pooled {
		if len(seg.Embedding) > 0 {
			pooledEmbeddings = append(pooledEmbeddings, seg.Embedding)
		}
	}
	if len(pooledEmbeddings) == 0 {
		return embeddings
	}
	return pooledEmbeddings
}

func (c *ColBERTScorer) segmentCacheStats(texts []string) (hits int, misses int) {
	if c == nil || c.cache == nil {
		return 0, len(texts)
	}
	for _, text := range texts {
		if c.cache.get("d\x00"+text) != nil {
			hits++
		} else {
			misses++
		}
	}
	return hits, len(texts) - hits
}

// embedTexts embeds retrieval texts in distinct query/document cache
// namespaces because task prefixes intentionally produce different vectors for
// identical raw text.
func (c *ColBERTScorer) embedTexts(ctx context.Context, texts []string, query bool) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	uncached := make([]string, 0, len(texts))
	uncachedKeys := make([]string, 0, len(texts))
	uncachedIdx := make([]int, 0, len(texts))
	prefix := "d\x00"
	if query {
		prefix = "q\x00"
	}

	for i, text := range texts {
		key := prefix + text
		if cached := c.cache.get(key); cached != nil {
			embeddings[i] = cached
		} else {
			uncached = append(uncached, text)
			uncachedKeys = append(uncachedKeys, key)
			uncachedIdx = append(uncachedIdx, i)
		}
	}

	if len(uncached) > 0 {
		var (
			newEmbs [][]float32
			err     error
		)
		if query {
			newEmbs, err = c.embedder.EmbedQueryBatch(ctx, uncached)
		} else {
			newEmbs, err = c.embedder.EmbedDocuments(ctx, uncached)
		}
		if err != nil {
			return nil, err
		}
		for i, idx := range uncachedIdx {
			embeddings[idx] = newEmbs[i]
			c.cache.set(uncachedKeys[i], newEmbs[i])
		}
	}

	return embeddings, nil
}

// decomposeQuery splits a query into meaningful terms/phrases.
// Focuses on extracting semantic units rather than individual words.
func decomposeQuery(query string) []string {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}

	var terms []string

	// First, add the full query as one term (captures full context)
	terms = append(terms, query)

	// Extract noun phrases and key terms
	words := tokenize(query)
	if len(words) <= 3 {
		// Short query: use individual meaningful words
		for _, w := range words {
			if len(w) > 2 && !isStopWord(w) {
				terms = append(terms, w)
			}
		}
	} else {
		// Longer query: extract bigrams and key terms
		for i := 0; i < len(words)-1; i++ {
			if !isStopWord(words[i]) || !isStopWord(words[i+1]) {
				bigram := words[i] + " " + words[i+1]
				terms = append(terms, bigram)
			}
		}
		// Also add individual key terms
		for _, w := range words {
			if len(w) > 3 && !isStopWord(w) {
				terms = append(terms, w)
			}
		}
	}

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, t := range terms {
		if !seen[t] {
			seen[t] = true
			unique = append(unique, t)
		}
	}

	// Limit to avoid too many embedding calls
	if len(unique) > 8 {
		unique = unique[:8]
	}

	return unique
}

// DecomposeDocument splits a document into meaningful segments.
// Uses sentence boundaries and code structure hints.
// Exported for use during indexing to pre-compute segment embeddings.
func DecomposeDocument(content string) []string {
	return decomposeDocumentWithBudget(content, legacyMaxDocumentSegments)
}

// DecomposeDocumentAdaptive splits a document using a raw-segment-aware sqrt(M)
// budget. Chunks at or below the legacy cap are left unchanged.
func DecomposeDocumentAdaptive(content string) []string {
	raw := DecomposeDocumentRaw(content)
	if len(raw) == 0 {
		return nil
	}
	return sampleSegmentsToBudget(raw, AdaptiveSegmentBudgetFromRawCount(len(raw)))
}

// DecomposeDocumentRaw splits a document into all natural segments without applying
// a representative budget cap. This is used by indexing paths that pool segments
// after embedding rather than sampling at the text level.
func DecomposeDocumentRaw(content string) []string {
	return decomposeDocumentWithBudget(content, 0)
}

// DecomposeDocumentWithMode selects legacy or adaptive document decomposition.
func DecomposeDocumentWithMode(content string, adaptive bool) []string {
	if adaptive {
		return DecomposeDocumentAdaptive(content)
	}
	return DecomposeDocument(content)
}

// AdaptiveSegmentBudget returns the adaptive representative budget for a
// document based on its raw segment count.
func AdaptiveSegmentBudget(content string) int {
	return AdaptiveSegmentBudgetFromRawCount(len(DecomposeDocumentRaw(content)))
}

// AdaptiveSegmentBudgetFromRawCount returns the compression-only adaptive budget
// for a chunk with the given number of raw segments. Chunks at or below the
// legacy cap are left unchanged; only over-cap chunks are pooled down.
func AdaptiveSegmentBudgetFromRawCount(rawCount int) int {
	if rawCount <= 0 {
		return 0
	}
	if rawCount <= legacyMaxDocumentSegments {
		return rawCount
	}

	budget := int(math.Ceil(math.Sqrt(float64(rawCount))))
	if budget < adaptiveMinDocumentSegments {
		budget = adaptiveMinDocumentSegments
	}
	if budget > adaptiveMaxDocumentSegments {
		budget = adaptiveMaxDocumentSegments
	}
	return budget
}

const (
	legacyMaxDocumentSegments   = 8
	adaptiveMinDocumentSegments = 3
	adaptiveMaxDocumentSegments = legacyMaxDocumentSegments
)

func (c *ColBERTScorer) documentSegments(content string) []string {
	return DecomposeDocumentWithMode(content, c != nil && c.adaptiveSegments)
}

// decomposeDocumentWithBudget splits a document into meaningful segments and caps
// the representative set to a caller-provided budget.
func decomposeDocumentWithBudget(content string, maxSegments int) []string {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}

	var segments []string

	// Split by newlines first (code structure)
	lines := strings.Split(content, "\n")

	var currentSegment strings.Builder
	currentLen := 0

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			// Empty line: flush current segment if substantial
			if currentLen > 20 {
				segments = append(segments, strings.TrimSpace(currentSegment.String()))
				currentSegment.Reset()
				currentLen = 0
			}
			continue
		}

		// Check if this line starts a new logical unit
		isNewUnit := strings.HasPrefix(line, "func ") ||
			strings.HasPrefix(line, "type ") ||
			strings.HasPrefix(line, "//") ||
			strings.HasPrefix(line, "def ") ||
			strings.HasPrefix(line, "class ") ||
			strings.HasPrefix(line, "#")

		if isNewUnit && currentLen > 20 {
			segments = append(segments, strings.TrimSpace(currentSegment.String()))
			currentSegment.Reset()
			currentLen = 0
		}

		currentSegment.WriteString(line)
		currentSegment.WriteString(" ")
		currentLen += len(line)

		// Flush if segment is getting long
		if currentLen > 200 {
			segments = append(segments, strings.TrimSpace(currentSegment.String()))
			currentSegment.Reset()
			currentLen = 0
		}
	}

	// Flush remaining
	if currentLen > 10 {
		segments = append(segments, strings.TrimSpace(currentSegment.String()))
	}

	return sampleSegmentsToBudget(segments, maxSegments)
}

func sampleSegmentsToBudget(segments []string, maxSegments int) []string {
	if maxSegments <= 0 || len(segments) <= maxSegments {
		return segments
	}
	// Keep first, last, and sample from the middle to preserve coverage.
	sampled := make([]string, 0, maxSegments)
	sampled = append(sampled, segments[0])
	middleBudget := maxSegments - 2
	if middleBudget <= 0 {
		return sampled
	}

	lastIdx := len(segments) - 1
	prevIdx := 0
	for i := 1; i <= middleBudget; i++ {
		idx := (i * lastIdx) / (middleBudget + 1)
		if idx <= prevIdx {
			idx = prevIdx + 1
		}
		if idx >= lastIdx {
			idx = lastIdx - 1
		}
		if idx <= prevIdx || idx >= lastIdx {
			break
		}
		sampled = append(sampled, segments[idx])
		prevIdx = idx
	}
	sampled = append(sampled, segments[len(segments)-1])
	return sampled
}

// tokenize splits text into lowercase tokens.
func tokenize(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(unicode.ToLower(r))
		} else if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

// isStopWord returns true for common English stop words.
func isStopWord(word string) bool {
	stops := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"to": true, "of": true, "in": true, "for": true, "on": true,
		"with": true, "at": true, "by": true, "from": true, "as": true,
		"into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "between": true,
		"and": true, "but": true, "or": true, "nor": true, "so": true,
		"yet": true, "both": true, "either": true, "neither": true,
		"not": true, "only": true, "own": true, "same": true, "than": true,
		"too": true, "very": true, "just": true, "also": true,
		"this": true, "that": true, "these": true, "those": true,
		"i": true, "me": true, "my": true, "we": true, "our": true,
		"you": true, "your": true, "he": true, "she": true, "it": true,
		"they": true, "them": true, "their": true, "what": true, "which": true,
		"who": true, "whom": true, "where": true, "when": true, "why": true, "how": true,
	}
	return stops[word]
}

// cosineSimilarity computes cosine similarity between two vectors.
// Deprecated: Use dotProductSimilarity for pre-normalized vectors (4x faster).
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// dotProductSimilarity computes similarity for PRE-NORMALIZED vectors.
// For unit vectors, dot product equals cosine similarity directly.
// Uses 8-way loop unrolling for better CPU pipeline utilization.
// This is ~4x faster than cosineSimilarity since no norm computation needed.
func dotProductSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	return util.DotProductUnrolled8(a, b)
}

// maxSimBatch computes MaxSim for a single query embedding against all doc embeddings.
// Uses pre-allocated distances buffer to avoid allocations in the hot path.
// Returns the maximum similarity found (-1 if no embeddings provided).
func maxSimBatch(qEmb []float32, docEmbs [][]float32, distances []float64) float64 {
	if len(docEmbs) == 0 {
		return -1
	}

	// Compute all similarities using unrolled dot product
	for i, dEmb := range docEmbs {
		distances[i] = util.DotProductUnrolled8(qEmb, dEmb)
	}

	// Find maximum
	maxSim := distances[0]
	for i := 1; i < len(docEmbs); i++ {
		if distances[i] > maxSim {
			maxSim = distances[i]
		}
	}
	return maxSim
}

// maxSimInt8 computes MaxSim for a query embedding against int8-quantized doc segments.
// Supports both int8 (quantized) and float32 (fallback for on-demand) embeddings.
// Returns the maximum similarity found (-1 if no segments provided).
func maxSimInt8(qEmb []float32, segments []store.ColBERTSegment) float64 {
	if len(segments) == 0 {
		return -1
	}

	maxSim := float64(-1)
	for _, seg := range segments {
		var sim float64
		if seg.EmbeddingInt8 != nil {
			// Use int8 quantized embedding (4x compressed storage)
			sim = util.DotProductInt8Unrolled8(qEmb, seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
		} else if seg.Embedding != nil {
			// Fall back to float32 (on-demand generated)
			sim = util.DotProductUnrolled8(qEmb, seg.Embedding)
		}
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

func prepareQueryTerms(queryEmbeddings [][]float32) []preparedQueryTerm {
	terms := make([]preparedQueryTerm, len(queryEmbeddings))
	for i, emb := range queryEmbeddings {
		terms[i] = preparedQueryTerm{
			embedding: emb,
			sum:       sumFloat32(emb),
		}
	}
	return terms
}

func prepareQueryTermsWithCodecs(queryEmbeddings [][]float32, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) []preparedQueryTerm {
	terms := prepareQueryTerms(queryEmbeddings)
	if tq != nil {
		for i := range terms {
			prepared, err := tq.PrepareQuery(terms[i].embedding)
			if err != nil {
				continue
			}
			terms[i].tqQuery = prepared
		}
	}
	if pq == nil || !pq.IsTrained() {
		return terms
	}

	for i := range terms {
		table, err := pq.PrecomputeQueryTable(terms[i].embedding)
		if err != nil {
			continue
		}
		terms[i].pqTable = table
	}

	return terms
}

func prepareQueryTermsWithPQ(queryEmbeddings [][]float32, pq *util.ProductQuantizer) []preparedQueryTerm {
	return prepareQueryTermsWithCodecs(queryEmbeddings, pq, nil)
}

func maxSimPrepared(term preparedQueryTerm, segments []store.ColBERTSegment, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) float64 {
	if tq != nil {
		if maxSim := maxSimPreparedTQMSE(term, segments, tq); maxSim >= 0 {
			return maxSim
		}
	}
	if pq != nil {
		if maxSim := maxSimPreparedPQ(term, segments); maxSim >= 0 {
			return maxSim
		}
	}
	return maxSimPreparedInt8(term, segments)
}

func maxSimPreparedTQMSE(term preparedQueryTerm, segments []store.ColBERTSegment, tq *util.TQMSEQuantizer) float64 {
	if len(segments) == 0 || tq == nil || !term.tqQuery.Valid() {
		return -1
	}

	maxSim := float64(-1)
	codeSize := tq.CodeSize()
	for _, seg := range segments {
		if len(seg.TQCodes) != codeSize {
			continue
		}
		sim := tq.Dot(term.tqQuery, util.TQMSECode{Codes: seg.TQCodes})
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

func maxSimPreparedInt8(term preparedQueryTerm, segments []store.ColBERTSegment) float64 {
	if len(segments) == 0 {
		return -1
	}

	maxSim := float64(-1)
	for _, seg := range segments {
		var sim float64
		if seg.EmbeddingInt8 != nil {
			sim = util.DotProductInt8AffinePrepared(term.embedding, seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin, term.sum)
		} else if seg.Embedding != nil {
			sim = util.DotProductUnrolled8(term.embedding, seg.Embedding)
		}
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

func maxSimPreparedPQ(term preparedQueryTerm, segments []store.ColBERTSegment) float64 {
	if len(segments) == 0 || len(term.pqTable) == 0 {
		return -1
	}

	maxSim := float64(-1)
	found := false
	for _, seg := range segments {
		if len(seg.PQCodes) == 0 {
			continue
		}
		found = true
		sim := 0.0
		for sub := range term.pqTable {
			sim += term.pqTable[sub][seg.PQCodes[sub]]
		}
		if sim > maxSim {
			maxSim = sim
		}
	}
	if !found {
		return -1
	}
	return maxSim
}

func hasPQSegments(segments []store.ColBERTSegment) bool {
	for _, seg := range segments {
		if len(seg.PQCodes) > 0 {
			return true
		}
	}
	return false
}

func exactColBERTScore(queryEmbeddings, docEmbeddings [][]float32) float64 {
	if len(queryEmbeddings) == 0 || len(docEmbeddings) == 0 {
		return 0
	}

	distances := make([]float64, len(docEmbeddings))
	var totalScore float64
	for _, qEmb := range queryEmbeddings {
		maxSim := maxSimBatch(qEmb, docEmbeddings, distances)
		if maxSim > 0 {
			totalScore += maxSim
		}
	}
	return totalScore / float64(len(queryEmbeddings))
}

func sumFloat32(values []float32) float64 {
	var sum float64
	for _, v := range values {
		sum += float64(v)
	}
	return sum
}
