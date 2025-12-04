package search

import (
	"context"
	"math"
	"strings"
	"sync"
	"unicode"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

// ColBERTScorer implements late interaction scoring similar to ColBERT.
// Instead of single-vector similarity, it computes MaxSim between query
// terms and document segments for more precise relevance scoring.
type ColBERTScorer struct {
	embedder *embed.Embedder
	cache    *segmentCache
}

// segmentCache caches segment embeddings to avoid recomputation.
type segmentCache struct {
	mu    sync.RWMutex
	items map[string][]float32 // segment text -> embedding
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
	c.items[key] = emb
}

// NewColBERTScorer creates a new ColBERT-style scorer.
func NewColBERTScorer(embedder *embed.Embedder) *ColBERTScorer {
	return &ColBERTScorer{
		embedder: embedder,
		cache:    newSegmentCache(1000),
	}
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
	docSegments := decomposeDocument(docContent)
	if len(docSegments) == 0 {
		return 0, nil
	}

	// Get embeddings for query terms
	queryEmbeddings, err := c.embedTexts(ctx, queryTerms)
	if err != nil {
		return 0, err
	}

	// Get embeddings for document segments
	docEmbeddings, err := c.embedTexts(ctx, docSegments)
	if err != nil {
		return 0, err
	}

	// Compute MaxSim: for each query term, find max similarity to any doc segment
	var totalScore float64
	for _, qEmb := range queryEmbeddings {
		maxSim := float64(-1)
		for _, dEmb := range docEmbeddings {
			sim := cosineSimilarity(qEmb, dEmb)
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
// Returns scores in the same order as documents.
func (c *ColBERTScorer) ScoreBatch(ctx context.Context, query string, documents []string) ([]float64, error) {
	// Decompose query into terms
	queryTerms := decomposeQuery(query)
	if len(queryTerms) == 0 {
		return make([]float64, len(documents)), nil
	}

	// Get query term embeddings (computed once)
	queryEmbeddings, err := c.embedTexts(ctx, queryTerms)
	if err != nil {
		return nil, err
	}

	util.Debugf(util.DebugDetailed, "ColBERT: %d query terms, scoring %d docs", len(queryTerms), len(documents))

	// Score each document
	scores := make([]float64, len(documents))
	for i, doc := range documents {
		docSegments := decomposeDocument(doc)
		if len(docSegments) == 0 {
			continue
		}

		docEmbeddings, err := c.embedTexts(ctx, docSegments)
		if err != nil {
			continue // Skip on error
		}

		// MaxSim computation
		var totalScore float64
		for _, qEmb := range queryEmbeddings {
			maxSim := float64(-1)
			for _, dEmb := range docEmbeddings {
				sim := cosineSimilarity(qEmb, dEmb)
				if sim > maxSim {
					maxSim = sim
				}
			}
			if maxSim > 0 {
				totalScore += maxSim
			}
		}
		scores[i] = totalScore / float64(len(queryTerms))
	}

	return scores, nil
}

// embedTexts embeds multiple texts, using cache where possible.
func (c *ColBERTScorer) embedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	var uncached []string
	var uncachedIdx []int

	// Check cache first
	for i, text := range texts {
		if cached := c.cache.get(text); cached != nil {
			embeddings[i] = cached
		} else {
			uncached = append(uncached, text)
			uncachedIdx = append(uncachedIdx, i)
		}
	}

	// Embed uncached texts
	if len(uncached) > 0 {
		newEmbs, err := c.embedder.EmbedBatch(ctx, uncached)
		if err != nil {
			return nil, err
		}
		for i, idx := range uncachedIdx {
			embeddings[idx] = newEmbs[i]
			c.cache.set(uncached[i], newEmbs[i])
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

// decomposeDocument splits a document into meaningful segments.
// Uses sentence boundaries and code structure hints.
func decomposeDocument(content string) []string {
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

	// Limit segments to avoid explosion
	if len(segments) > 10 {
		// Keep first, last, and sample from middle
		sampled := make([]string, 0, 10)
		sampled = append(sampled, segments[0])
		step := len(segments) / 8
		if step < 1 {
			step = 1
		}
		for i := step; i < len(segments)-1; i += step {
			sampled = append(sampled, segments[i])
			if len(sampled) >= 9 {
				break
			}
		}
		sampled = append(sampled, segments[len(segments)-1])
		segments = sampled
	}

	return segments
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
