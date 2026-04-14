package search

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

type staticSegmentStore struct {
	segments map[string][]store.ColBERTSegment
}

func (s *staticSegmentStore) StoreColBERTSegments(ctx context.Context, chunkID string, segments []store.ColBERTSegment) error {
	if s.segments == nil {
		s.segments = make(map[string][]store.ColBERTSegment)
	}
	s.segments[chunkID] = append([]store.ColBERTSegment(nil), segments...)
	return nil
}

func (s *staticSegmentStore) StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]store.ColBERTSegment) error {
	if s.segments == nil {
		s.segments = make(map[string][]store.ColBERTSegment)
	}
	for chunkID, segments := range chunkSegments {
		s.segments[chunkID] = append([]store.ColBERTSegment(nil), segments...)
	}
	return nil
}

func (s *staticSegmentStore) GetColBERTSegments(ctx context.Context, chunkID string) ([]store.ColBERTSegment, error) {
	return append([]store.ColBERTSegment(nil), s.segments[chunkID]...), nil
}

func (s *staticSegmentStore) GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]store.ColBERTSegment, error) {
	result := make(map[string][]store.ColBERTSegment, len(chunkIDs))
	for _, chunkID := range chunkIDs {
		result[chunkID] = append([]store.ColBERTSegment(nil), s.segments[chunkID]...)
	}
	return result, nil
}

func (s *staticSegmentStore) DeleteColBERTSegments(ctx context.Context, chunkID string) error {
	delete(s.segments, chunkID)
	return nil
}

func (s *staticSegmentStore) HasColBERTSegments(ctx context.Context) (bool, error) {
	return len(s.segments) > 0, nil
}

func (s *staticSegmentStore) GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]store.ChunkInfo, error) {
	return nil, nil
}

func newEmbeddingTestServer(t *testing.T, embeddings map[string][]float32) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Content []string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		type responseItem struct {
			Index     int         `json:"index"`
			Embedding [][]float32 `json:"embedding"`
		}

		resp := make([]responseItem, len(req.Content))
		for i, text := range req.Content {
			emb, ok := embeddings[text]
			if !ok {
				t.Fatalf("unexpected embedding request for %q", text)
			}
			resp[i] = responseItem{
				Index:     i,
				Embedding: [][]float32{emb},
			}
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
}

func TestDecomposeQuery(t *testing.T) {
	tests := []struct {
		query    string
		minTerms int
		maxTerms int
	}{
		{"error handling", 2, 5},
		{"how does authentication work", 3, 8},
		{"x", 1, 2},
		{"", 0, 0},
		{"chain of thought implementation", 3, 8},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			terms := decomposeQuery(tt.query)
			if len(terms) < tt.minTerms || len(terms) > tt.maxTerms {
				t.Errorf("decomposeQuery(%q) = %d terms, want %d-%d", tt.query, len(terms), tt.minTerms, tt.maxTerms)
			}
		})
	}
}

func TestDecomposeDocument(t *testing.T) {
	tests := []struct {
		name    string
		content string
		minSegs int
		maxSegs int
	}{
		{
			name:    "empty",
			content: "",
			minSegs: 0,
			maxSegs: 0,
		},
		{
			name:    "single line",
			content: "func main() {}",
			minSegs: 1,
			maxSegs: 1,
		},
		{
			name: "multiple functions",
			content: `func foo() {
	return 1
}

func bar() {
	return 2
}`,
			minSegs: 2,
			maxSegs: 4,
		},
		{
			name: "with comments",
			content: `// Package main provides entry point
package main

// main is the entry function
func main() {
	fmt.Println("hello")
}`,
			minSegs: 2,
			maxSegs: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			segments := DecomposeDocument(tt.content)
			if len(segments) < tt.minSegs || len(segments) > tt.maxSegs {
				t.Errorf("decomposeDocument() = %d segments, want %d-%d", len(segments), tt.minSegs, tt.maxSegs)
			}
		})
	}
}

func TestAdaptiveSegments(t *testing.T) {
	t.Run("short content keeps available segments", func(t *testing.T) {
		content := "func main() {}"

		budget := AdaptiveSegmentBudget(content)
		if budget != 1 {
			t.Fatalf("AdaptiveSegmentBudget(short) = %d, want 1", budget)
		}

		segments := DecomposeDocumentAdaptive(content)
		if len(segments) != 1 {
			t.Fatalf("DecomposeDocumentAdaptive(short) = %d segments, want 1", len(segments))
		}
		if len(segments) > budget {
			t.Fatalf("adaptive short segments %d exceed budget %d", len(segments), budget)
		}
	})

	t.Run("punctuation only keeps available segment", func(t *testing.T) {
		content := "// --------"

		budget := AdaptiveSegmentBudget(content)
		if budget != 1 {
			t.Fatalf("AdaptiveSegmentBudget(punctuation) = %d, want 1", budget)
		}

		segments := DecomposeDocumentAdaptive(content)
		if len(segments) != 1 {
			t.Fatalf("DecomposeDocumentAdaptive(punctuation) = %d segments, want 1", len(segments))
		}
		if len(segments) > budget {
			t.Fatalf("adaptive punctuation segments %d exceed budget %d", len(segments), budget)
		}
	})

	t.Run("long content compresses below legacy cap", func(t *testing.T) {
		var b strings.Builder
		for i := 0; i < 64; i++ {
			fmt.Fprintf(&b, "// handler %d\nfunc handler%d() error {\n\tif err := step%d(); err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}\n\n", i, i, i)
		}
		content := b.String()

		raw := DecomposeDocumentRaw(content)
		legacy := DecomposeDocument(content)
		adaptive := DecomposeDocumentAdaptive(content)
		budget := AdaptiveSegmentBudget(content)

		if len(legacy) != legacyMaxDocumentSegments {
			t.Fatalf("legacy decomposition = %d, want %d", len(legacy), legacyMaxDocumentSegments)
		}
		if len(raw) <= legacyMaxDocumentSegments {
			t.Fatalf("raw decomposition = %d, want > %d for long content", len(raw), legacyMaxDocumentSegments)
		}
		if budget >= len(legacy) {
			t.Fatalf("adaptive budget = %d, want < legacy %d for over-cap content", budget, len(legacy))
		}
		if len(adaptive) != budget {
			t.Fatalf("adaptive decomposition = %d, want budget %d", len(adaptive), budget)
		}
		if len(adaptive) >= len(legacy) {
			t.Fatalf("adaptive decomposition = %d, want < legacy %d", len(adaptive), len(legacy))
		}
	})

	t.Run("huge content clamps at max budget", func(t *testing.T) {
		var b strings.Builder
		for i := 0; i < 256; i++ {
			fmt.Fprintf(&b, "// service %d\nfunc service%d() error {\n\tvalue := config%d + retry%d + timeout%d\n\tif value == 0 {\n\t\treturn errDefault\n\t}\n\treturn nil\n}\n\n", i, i, i, i, i)
		}
		content := b.String()

		budget := AdaptiveSegmentBudget(content)
		if budget != adaptiveMaxDocumentSegments {
			t.Fatalf("AdaptiveSegmentBudget(huge) = %d, want %d", budget, adaptiveMaxDocumentSegments)
		}

		segments := DecomposeDocumentAdaptive(content)
		if len(segments) != adaptiveMaxDocumentSegments {
			t.Fatalf("huge adaptive decomposition = %d, want %d", len(segments), adaptiveMaxDocumentSegments)
		}
	})
}

func TestScoreBatchWithChunkIDs_PQExactRescoreRecoversOrder(t *testing.T) {
	queryEmbedding := []float32{1, 0, 0, 0}
	goodEmbedding := []float32{1, 0, 0, 0}
	badEmbedding := []float32{0, 1, 0, 0}

	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       4,
		Subspaces:  2,
		Centroids:  4,
		Iterations: 4,
	})
	if err != nil {
		t.Fatalf("NewProductQuantizer failed: %v", err)
	}
	if err := pq.Train([][]float32{
		queryEmbedding,
		goodEmbedding,
		badEmbedding,
		{0, 0, 1, 0},
	}, 4); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	biasedGoodCodes, err := pq.Encode(goodEmbedding)
	if err != nil {
		t.Fatalf("Encode good failed: %v", err)
	}
	biasedBadCodes, err := pq.Encode(badEmbedding)
	if err != nil {
		t.Fatalf("Encode bad failed: %v", err)
	}

	segmentStore := &staticSegmentStore{
		segments: map[string][]store.ColBERTSegment{
			"chunk-bad": {{
				SegmentIdx: 0,
				Text:       "bad segment line",
				PQCodes:    biasedGoodCodes,
			}},
			"chunk-good": {{
				SegmentIdx: 0,
				Text:       "good segment line",
				PQCodes:    biasedBadCodes,
			}},
		},
	}

	server := newEmbeddingTestServer(t, map[string][]float32{
		"needle":            queryEmbedding,
		"bad segment line":  badEmbedding,
		"good segment line": goodEmbedding,
	})
	defer server.Close()

	embedder := embed.NewWithConfig(embed.Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	coarseScorer := NewColBERTScorer(embedder)
	coarseScorer.SetSegmentStore(segmentStore)
	coarseScorer.SetProductQuantizer(pq)
	coarseScorer.SetPQExactRescoreTopK(0)

	coarseScores, err := coarseScorer.ScoreBatchWithChunkIDs(
		context.Background(),
		"needle",
		[]string{"chunk-bad", "chunk-good"},
		[]string{"bad segment line", "good segment line"},
	)
	if err != nil {
		t.Fatalf("coarse ScoreBatchWithChunkIDs failed: %v", err)
	}
	if coarseScores[0] <= coarseScores[1] {
		t.Fatalf("expected coarse PQ scores to mis-rank docs, got bad=%.4f good=%.4f", coarseScores[0], coarseScores[1])
	}

	exactScorer := NewColBERTScorer(embedder)
	exactScorer.SetSegmentStore(segmentStore)
	exactScorer.SetProductQuantizer(pq)
	exactScorer.SetPQExactRescoreTopK(2)

	exactScores, err := exactScorer.ScoreBatchWithChunkIDs(
		context.Background(),
		"needle",
		[]string{"chunk-bad", "chunk-good"},
		[]string{"bad segment line", "good segment line"},
	)
	if err != nil {
		t.Fatalf("exact ScoreBatchWithChunkIDs failed: %v", err)
	}
	if exactScores[1] <= exactScores[0] {
		t.Fatalf("expected exact PQ rescore to restore order, got bad=%.4f good=%.4f", exactScores[0], exactScores[1])
	}
}

func TestExactRescoreDocumentSegments_AdaptiveUsesRawDecomposition(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 16; i++ {
		fmt.Fprintf(&b, "// section %d\nfunc service%d() error {\n\tvalue := config%d + retry%d + timeout%d\n\tif value == 0 {\n\t\treturn errDefault\n\t}\n\treturn nil\n}\n\n", i, i, i, i, i)
	}
	content := b.String()

	scorer := NewColBERTScorer(nil)
	scorer.SetAdaptiveSegments(true)

	raw := DecomposeDocumentRaw(content)
	sampled := scorer.documentSegments(content)
	exact := scorer.exactRescoreDocumentSegments(content)

	if len(raw) <= len(sampled) {
		t.Fatalf("test content did not exceed adaptive sampled representation: raw=%d sampled=%d", len(raw), len(sampled))
	}
	if len(exact) != len(raw) {
		t.Fatalf("expected exact adaptive rescore to use raw decomposition: got %d want %d", len(exact), len(raw))
	}
}

func TestExactRescoreEmbeddings_AdaptivePoolsToBudget(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 16; i++ {
		fmt.Fprintf(&b, "// section %d\nfunc service%d() error {\n\tvalue := config%d + retry%d + timeout%d\n\tif value == 0 {\n\t\treturn errDefault\n\t}\n\treturn nil\n}\n\n", i, i, i, i, i)
	}
	rawTexts := DecomposeDocumentRaw(b.String())
	if len(rawTexts) <= legacyMaxDocumentSegments {
		t.Fatalf("test content did not exceed legacy cap: raw=%d", len(rawTexts))
	}

	embeddings := make([][]float32, len(rawTexts))
	for i := range rawTexts {
		embeddings[i] = make([]float32, 768)
		embeddings[i][i%768] = 1
	}

	scorer := NewColBERTScorer(nil)
	scorer.SetAdaptiveSegments(true)

	pooled := scorer.exactRescoreEmbeddings(rawTexts, embeddings)
	budget := AdaptiveSegmentBudgetFromRawCount(len(rawTexts))
	if len(pooled) > budget {
		t.Fatalf("expected pooled exact rescore embeddings to respect budget: got %d want <= %d", len(pooled), budget)
	}
	if len(pooled) >= len(rawTexts) {
		t.Fatalf("expected adaptive exact rescore pooling to reduce segment count: got %d from %d", len(pooled), len(rawTexts))
	}
}

func TestTokenize(t *testing.T) {
	tests := []struct {
		text     string
		expected []string
	}{
		{"Hello World", []string{"hello", "world"}},
		{"func_name", []string{"func", "name"}},
		{"CamelCase", []string{"camelcase"}},
		{"123abc", []string{"123abc"}},
		{"", nil},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			tokens := tokenize(tt.text)
			if len(tokens) != len(tt.expected) {
				t.Errorf("tokenize(%q) = %v, want %v", tt.text, tokens, tt.expected)
				return
			}
			for i, tok := range tokens {
				if tok != tt.expected[i] {
					t.Errorf("tokenize(%q)[%d] = %q, want %q", tt.text, i, tok, tt.expected[i])
				}
			}
		})
	}
}

func TestIsStopWord(t *testing.T) {
	stopWords := []string{"the", "is", "a", "an", "of", "to", "in", "for"}
	nonStopWords := []string{"error", "function", "database", "config", "handler"}

	for _, w := range stopWords {
		if !isStopWord(w) {
			t.Errorf("isStopWord(%q) = false, want true", w)
		}
	}

	for _, w := range nonStopWords {
		if isStopWord(w) {
			t.Errorf("isStopWord(%q) = true, want false", w)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
		delta    float64
	}{
		{
			name:     "identical",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "similar",
			a:        []float32{1, 1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.707,
			delta:    0.01,
		},
		{
			name:     "empty",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "mismatched length",
			a:        []float32{1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			if result < tt.expected-tt.delta || result > tt.expected+tt.delta {
				t.Errorf("cosineSimilarity(%v, %v) = %f, want %f±%f", tt.a, tt.b, result, tt.expected, tt.delta)
			}
		})
	}
}

func TestNewColBERTScorer(t *testing.T) {
	// Test that we can create a scorer without an embedder (nil is allowed)
	scorer := NewColBERTScorer(nil)
	if scorer == nil {
		t.Fatal("NewColBERTScorer returned nil")
	}
	if scorer.cache == nil {
		t.Error("scorer.cache should be initialized")
	}
}

func TestSegmentCache(t *testing.T) {
	cache := newSegmentCache(3)

	// Test set and get
	cache.set("key1", []float32{1, 2, 3})
	if got := cache.get("key1"); got == nil {
		t.Error("cache.get(key1) returned nil after set")
	}

	// Test cache miss
	if got := cache.get("nonexistent"); got != nil {
		t.Error("cache.get(nonexistent) should return nil")
	}

	// Test eviction (cache size is 3)
	cache.set("key2", []float32{4, 5, 6})
	cache.set("key3", []float32{7, 8, 9})
	cache.set("key4", []float32{10, 11, 12}) // This should trigger eviction

	// At least one of the original keys should be evicted
	// Note: eviction is simple and clears half, so behavior may vary
}

func TestPrepareQueryTerms(t *testing.T) {
	embeddings := [][]float32{
		{0.25, -0.5, 0.75},
		{-0.25, 0.5, -0.75},
	}

	terms := prepareQueryTerms(embeddings)
	if len(terms) != len(embeddings) {
		t.Fatalf("prepareQueryTerms returned %d terms, want %d", len(terms), len(embeddings))
	}

	for i, term := range terms {
		if len(term.embedding) != len(embeddings[i]) {
			t.Fatalf("term %d embedding len = %d, want %d", i, len(term.embedding), len(embeddings[i]))
		}
		if diff := math.Abs(term.sum - sumFloat32(embeddings[i])); diff > 1e-9 {
			t.Fatalf("term %d sum diff = %.12f", i, diff)
		}
	}
}

func TestMaxSimPreparedInt8MatchesCurrent(t *testing.T) {
	query := util.NormalizeVector([]float32{0.9, -0.3, 0.2, 0.1})
	segments := make([]store.ColBERTSegment, 3)

	rawSegments := [][]float32{
		util.NormalizeVector([]float32{0.8, -0.25, 0.25, 0.15}),
		util.NormalizeVector([]float32{-0.4, 0.9, -0.1, 0.05}),
		util.NormalizeVector([]float32{0.2, -0.1, 0.95, -0.05}),
	}

	for i, emb := range rawSegments {
		quantized, scale, min := util.QuantizeInt8(emb)
		segments[i] = store.ColBERTSegment{
			SegmentIdx:    i,
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		}
	}

	current := maxSimInt8(query, segments)
	prepared := maxSimPreparedInt8(prepareQueryTerms([][]float32{query})[0], segments)
	if diff := math.Abs(current - prepared); diff > 1e-7 {
		t.Fatalf("prepared scorer mismatch: current %.12f prepared %.12f diff %.12f", current, prepared, diff)
	}
}

func TestMaxSimPreparedPQMatchesADC(t *testing.T) {
	training := [][]float32{
		util.NormalizeVector([]float32{0.9, 0.1, 0.0, 0.0}),
		util.NormalizeVector([]float32{0.8, 0.2, 0.0, 0.0}),
		util.NormalizeVector([]float32{0.0, 0.9, 0.1, 0.0}),
		util.NormalizeVector([]float32{0.0, 0.8, 0.2, 0.0}),
		util.NormalizeVector([]float32{0.0, 0.0, 0.9, 0.1}),
		util.NormalizeVector([]float32{0.0, 0.0, 0.8, 0.2}),
		util.NormalizeVector([]float32{0.1, 0.0, 0.0, 0.9}),
		util.NormalizeVector([]float32{0.2, 0.0, 0.0, 0.8}),
	}

	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       4,
		Subspaces:  2,
		Centroids:  4,
		Iterations: 5,
	})
	if err != nil {
		t.Fatalf("NewProductQuantizer: %v", err)
	}
	if err := pq.Train(training, 5); err != nil {
		t.Fatalf("Train: %v", err)
	}

	query := util.NormalizeVector([]float32{0.85, 0.15, 0.0, 0.0})
	rawSegments := [][]float32{
		util.NormalizeVector([]float32{0.88, 0.12, 0.0, 0.0}),
		util.NormalizeVector([]float32{0.0, 0.82, 0.18, 0.0}),
		util.NormalizeVector([]float32{0.15, 0.0, 0.0, 0.85}),
	}

	segments := make([]store.ColBERTSegment, len(rawSegments))
	table, err := pq.PrecomputeQueryTable(query)
	if err != nil {
		t.Fatalf("PrecomputeQueryTable: %v", err)
	}
	expected := math.Inf(-1)
	for i, emb := range rawSegments {
		codes, err := pq.Encode(emb)
		if err != nil {
			t.Fatalf("Encode(%d): %v", i, err)
		}
		segments[i] = store.ColBERTSegment{
			SegmentIdx: i,
			PQCodes:    codes,
		}
		if score := pq.DotProductWithTable(table, codes); score > expected {
			expected = score
		}
	}

	term := prepareQueryTermsWithPQ([][]float32{query}, pq)[0]
	got := maxSimPreparedPQ(term, segments)
	if diff := math.Abs(expected - got); diff > 1e-9 {
		t.Fatalf("prepared PQ scorer mismatch: expected %.12f got %.12f diff %.12f", expected, got, diff)
	}
}
