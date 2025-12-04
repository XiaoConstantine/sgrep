package search

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

type mockStore struct {
	docs      []*store.Document
	distances []float64
	searchErr error
	mu        sync.Mutex
}

func (m *mockStore) Store(ctx context.Context, doc *store.Document) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.docs = append(m.docs, doc)
	return nil
}

func (m *mockStore) StoreBatch(ctx context.Context, docs []*store.Document) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.docs = append(m.docs, docs...)
	return nil
}

func (m *mockStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*store.Document, []float64, error) {
	if m.searchErr != nil {
		return nil, nil, m.searchErr
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	n := len(m.docs)
	if limit < n {
		n = limit
	}

	results := make([]*store.Document, n)
	dists := make([]float64, n)

	for i := 0; i < n; i++ {
		results[i] = m.docs[i]
		if i < len(m.distances) {
			dists[i] = m.distances[i]
		} else {
			dists[i] = float64(i) * 0.1
		}
	}

	return results, dists, nil
}

func (m *mockStore) Stats(ctx context.Context) (*store.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return &store.Stats{Chunks: int64(len(m.docs))}, nil
}

func (m *mockStore) DeleteByPath(ctx context.Context, filepath string) error {
	return nil
}

func (m *mockStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*store.Document, []float64, error) {
	// For mock, just delegate to Search
	return m.Search(ctx, embedding, limit, threshold)
}

func (m *mockStore) Close() error {
	return nil
}

func TestNew(t *testing.T) {
	ms := &mockStore{}
	s := New(ms)

	if s == nil {
		t.Fatal("New should return non-nil searcher")
	}
	if s.store != ms {
		t.Error("store should be set")
	}
	if s.queryCache == nil {
		t.Error("queryCache should be initialized")
	}
}

func TestNewWithOptions(t *testing.T) {
	ms := &mockStore{}
	s := NewWithOptions(ms, 50, 2*time.Minute)

	if s == nil {
		t.Fatal("NewWithOptions should return non-nil searcher")
	}
}

func TestNewWithConfig(t *testing.T) {
	ms := &mockStore{}
	eb := util.NewEventBox()

	cfg := Config{
		Store:     ms,
		CacheSize: 200,
		CacheTTL:  10 * time.Minute,
		EventBox:  eb,
	}

	s := NewWithConfig(cfg)

	if s == nil {
		t.Fatal("NewWithConfig should return non-nil searcher")
	}
	if s.eventBox != eb {
		t.Error("eventBox should be set")
	}
}

func TestSearcher_Search_EmptyStore(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{}
	s := New(ms)

	results, err := s.Search(context.Background(), "test query", 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results from empty store, got %d", len(results))
	}
}

func TestSearcher_Search_WithResults(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "func main()", StartLine: 1, EndLine: 5},
			{ID: "doc2", FilePath: "/test2.go", Content: "type Foo struct{}", StartLine: 10, EndLine: 15},
		},
		distances: []float64{0.5, 1.2},
	}
	s := New(ms)

	results, err := s.Search(context.Background(), "main function", 10, 2.0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	if results[0].FilePath != "/test.go" {
		t.Errorf("expected first result filepath /test.go, got %s", results[0].FilePath)
	}
	// Score is boosted by BoostImpl factor (0.92) for non-test files: 0.5 * 0.92 = 0.46
	expectedScore := 0.5 * 0.92
	if results[0].Score != expectedScore {
		t.Errorf("expected score %f (0.5 * 0.92 boost), got %f", expectedScore, results[0].Score)
	}
}

func TestSearcher_Search_StoreError(t *testing.T) {
	ms := &mockStore{
		searchErr: errors.New("store error"),
	}
	s := New(ms)

	_, err := s.Search(context.Background(), "test query", 10, 2.0)
	if err == nil {
		t.Error("expected error from store to propagate")
	}
}

func TestSearcher_Search_Caching(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}

	s := New(ms)

	_, _ = s.Search(context.Background(), "test query", 10, 2.0)
	_, _ = s.Search(context.Background(), "test query", 10, 2.0)

	stats := s.CacheStats()
	if stats.Size == 0 {
		t.Error("expected cache to have entries after search")
	}
}

func TestSearcher_Search_DifferentParams(t *testing.T) {
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	r1, _ := s.Search(context.Background(), "test", 5, 1.0)
	r2, _ := s.Search(context.Background(), "test", 10, 1.0)
	r3, _ := s.Search(context.Background(), "test", 5, 2.0)

	_ = r1
	_ = r2
	_ = r3
}

func TestSearcher_Search_EventBox(t *testing.T) {
	eb := util.NewEventBox()
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}

	s := NewWithConfig(Config{
		Store:    ms,
		EventBox: eb,
	})

	done := make(chan bool)
	go func() {
		evtType, _ := eb.Wait(util.EvtSearchStart, util.EvtSearchComplete)
		if evtType == util.EvtSearchStart || evtType == util.EvtSearchComplete {
			done <- true
		}
	}()

	go func() {
		_, _ = s.Search(context.Background(), "test query", 10, 2.0)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("expected search events to be emitted")
	}
}

func TestSearcher_ClearCache(t *testing.T) {
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	_, _ = s.Search(context.Background(), "test query", 10, 2.0)
	s.ClearCache()

	stats := s.CacheStats()
	if stats.Size != 0 {
		t.Errorf("expected cache size 0 after clear, got %d", stats.Size)
	}
}

func TestSearcher_CacheStats(t *testing.T) {
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	stats := s.CacheStats()
	if stats.Capacity != 100 {
		t.Errorf("expected default capacity 100, got %d", stats.Capacity)
	}
}

func TestSearcher_cacheKey(t *testing.T) {
	ms := &mockStore{}
	s := New(ms)

	key1 := s.cacheKey("query", 10, 2.0)
	key2 := s.cacheKey("query", 10, 2.0)
	key3 := s.cacheKey("query", 5, 2.0)
	key4 := s.cacheKey("different", 10, 2.0)

	if key1 != key2 {
		t.Error("same params should produce same cache key")
	}
	if key1 == key3 {
		t.Error("different limit should produce different cache key")
	}
	if key1 == key4 {
		t.Error("different query should produce different cache key")
	}
}

func TestResult_Fields(t *testing.T) {
	r := Result{
		FilePath:  "/test/file.go",
		StartLine: 10,
		EndLine:   20,
		Score:     0.5,
		Content:   "func main() {}",
	}

	if r.FilePath != "/test/file.go" {
		t.Errorf("unexpected FilePath: %s", r.FilePath)
	}
	if r.StartLine != 10 {
		t.Errorf("unexpected StartLine: %d", r.StartLine)
	}
	if r.EndLine != 20 {
		t.Errorf("unexpected EndLine: %d", r.EndLine)
	}
	if r.Score != 0.5 {
		t.Errorf("unexpected Score: %f", r.Score)
	}
	if r.Content != "func main() {}" {
		t.Errorf("unexpected Content: %s", r.Content)
	}
}

func TestSearcher_ConcurrentSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := s.Search(context.Background(), "query"+string(rune('0'+i%10)), 10, 2.0)
			if err != nil {
				t.Errorf("concurrent search failed: %v", err)
			}
		}(i)
	}
	wg.Wait()
}

// Benchmarks

func BenchmarkSearcher_Search_CacheHit(b *testing.B) {
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	_, _ = s.Search(context.Background(), "test query", 10, 2.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = s.Search(context.Background(), "test query", 10, 2.0)
	}
}

func BenchmarkSearcher_Search_CacheMiss(b *testing.B) {
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/test.go", Content: "content"},
		},
	}
	s := New(ms)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = s.Search(context.Background(), "query"+string(rune(i%256)), 10, 2.0)
	}
}

func BenchmarkSearcher_cacheKey(b *testing.B) {
	ms := &mockStore{}
	s := New(ms)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.cacheKey("test query", 10, 2.0)
	}
}

func TestDefaultSearchOptions(t *testing.T) {
	opts := DefaultSearchOptions()
	if opts.Limit != 10 {
		t.Errorf("expected Limit 10, got %d", opts.Limit)
	}
	if opts.Threshold != 0.65 {
		t.Errorf("expected Threshold 0.65, got %f", opts.Threshold)
	}
	if opts.IncludeTests != false {
		t.Error("expected IncludeTests false")
	}
	if opts.Deduplicate != true {
		t.Error("expected Deduplicate true")
	}
	if opts.BoostImpl != 0.92 {
		t.Errorf("expected BoostImpl 0.85, got %f", opts.BoostImpl)
	}
}

func TestSearcher_SearchWithOptions_IncludeTests(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{
		docs: []*store.Document{
			{ID: "impl", FilePath: "/main.go", Content: "func main()", IsTest: false},
			{ID: "test", FilePath: "/main_test.go", Content: "func TestMain()", IsTest: true},
		},
		distances: []float64{0.5, 0.6},
	}
	s := New(ms)

	// Without tests
	opts := DefaultSearchOptions()
	opts.IncludeTests = false
	results, err := s.SearchWithOptions(context.Background(), "main", opts)
	if err != nil {
		t.Fatal(err)
	}

	for _, r := range results {
		if r.IsTest {
			t.Error("should not include test files when IncludeTests=false")
		}
	}
}

func TestSearcher_SearchWithOptions_Deduplicate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	// Smart deduplication: keeps non-overlapping chunks, collapses overlapping ones
	ms := &mockStore{
		docs: []*store.Document{
			{ID: "chunk1", FilePath: "/main.go", Content: "func foo()", StartLine: 1, EndLine: 5},
			{ID: "chunk2", FilePath: "/main.go", Content: "func bar()", StartLine: 10, EndLine: 15}, // non-overlapping with chunk1
			{ID: "chunk3", FilePath: "/main.go", Content: "func foo2()", StartLine: 2, EndLine: 6}, // overlaps with chunk1
			{ID: "chunk4", FilePath: "/other.go", Content: "func baz()", StartLine: 1, EndLine: 5},
		},
		distances: []float64{0.3, 0.5, 0.6, 0.4},
	}
	s := New(ms)

	// With deduplication
	opts := DefaultSearchOptions()
	opts.Deduplicate = true
	results, err := s.SearchWithOptions(context.Background(), "func", opts)
	if err != nil {
		t.Fatal(err)
	}

	// Smart dedup keeps non-overlapping chunks from same file
	// chunk1 (1-5) and chunk2 (10-15) are non-overlapping -> keep both
	// chunk3 (2-6) overlaps with chunk1 (1-5) -> collapsed
	// So main.go should have 2 chunks, other.go should have 1
	fileCount := make(map[string]int)
	for _, r := range results {
		fileCount[r.FilePath]++
	}

	if fileCount["/main.go"] != 2 {
		t.Errorf("main.go should have 2 non-overlapping chunks, got %d", fileCount["/main.go"])
	}
	if fileCount["/other.go"] != 1 {
		t.Errorf("other.go should have 1 chunk, got %d", fileCount["/other.go"])
	}
}

func TestSearcher_SearchWithOptions_NoBoost(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test that requires embedding server")
	}

	ms := &mockStore{
		docs: []*store.Document{
			{ID: "doc1", FilePath: "/main.go", Content: "content", IsTest: false},
		},
		distances: []float64{1.0},
	}
	s := New(ms)

	// With boost disabled (BoostImpl = 1.0 means no boost)
	opts := DefaultSearchOptions()
	opts.BoostImpl = 1.0
	results, err := s.SearchWithOptions(context.Background(), "test", opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) > 0 && results[0].Score != 1.0 {
		t.Errorf("expected score 1.0 with no boost, got %f", results[0].Score)
	}
}

func TestSearcher_cacheKeyWithOpts(t *testing.T) {
	ms := &mockStore{}
	s := New(ms)

	opts1 := DefaultSearchOptions()
	opts2 := DefaultSearchOptions()
	opts2.IncludeTests = true

	key1 := s.cacheKeyWithOpts("query", opts1)
	key2 := s.cacheKeyWithOpts("query", opts2)

	if key1 == key2 {
		t.Error("different options should produce different cache keys")
	}

	// Same options should produce same key
	key3 := s.cacheKeyWithOpts("query", opts1)
	if key1 != key3 {
		t.Error("same options should produce same cache key")
	}
}

func TestDeduplicateResults(t *testing.T) {
	// Overlapping chunks from same file should be deduplicated
	results := []Result{
		{FilePath: "/a.go", StartLine: 1, EndLine: 10, Score: 0.3},
		{FilePath: "/a.go", StartLine: 5, EndLine: 15, Score: 0.5}, // Overlaps with first
		{FilePath: "/b.go", StartLine: 1, EndLine: 10, Score: 0.4},
		{FilePath: "/a.go", StartLine: 1, EndLine: 10, Score: 0.2}, // Same range, best score
	}

	deduped := deduplicateResults(results)

	if len(deduped) != 2 {
		t.Errorf("expected 2 deduplicated results (overlapping chunks collapsed), got %d", len(deduped))
	}

	// Check that /a.go kept the best score
	for _, r := range deduped {
		if r.FilePath == "/a.go" && r.Score != 0.2 {
			t.Errorf("expected best score 0.2 for /a.go, got %f", r.Score)
		}
	}
}

func TestDeduplicateResults_KeepsNonOverlapping(t *testing.T) {
	// Non-overlapping chunks from same file should be kept
	results := []Result{
		{FilePath: "/a.go", StartLine: 1, EndLine: 10, Score: 0.3},
		{FilePath: "/a.go", StartLine: 50, EndLine: 60, Score: 0.4}, // Different part of file
		{FilePath: "/a.go", StartLine: 100, EndLine: 110, Score: 0.5}, // Another part
		{FilePath: "/b.go", StartLine: 1, EndLine: 10, Score: 0.2},
	}

	deduped := deduplicateResults(results)

	if len(deduped) != 4 {
		t.Errorf("expected 4 results (non-overlapping chunks kept), got %d", len(deduped))
	}

	// Count chunks from /a.go
	aCount := 0
	for _, r := range deduped {
		if r.FilePath == "/a.go" {
			aCount++
		}
	}
	if aCount != 3 {
		t.Errorf("expected 3 chunks from /a.go (non-overlapping), got %d", aCount)
	}
}

func TestDeduplicateResults_Empty(t *testing.T) {
	results := deduplicateResults(nil)
	if len(results) != 0 {
		t.Error("expected empty result for nil input")
	}

	results = deduplicateResults([]Result{})
	if len(results) != 0 {
		t.Error("expected empty result for empty input")
	}
}

func TestCanonicalPath(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "regular path unchanged",
			input:    "pkg/search/search.go",
			expected: "pkg/search/search.go",
		},
		{
			name:     "worktree path normalized",
			input:    ".worktrees/feature-branch/pkg/search/search.go",
			expected: "pkg/search/search.go",
		},
		{
			name:     "worktree with hyphen in name",
			input:    ".worktrees/a2a-integration/pkg/core/signature.go",
			expected: "pkg/core/signature.go",
		},
		{
			name:     "nested worktree path",
			input:    "some/prefix/.worktrees/branch/pkg/foo.go",
			expected: "pkg/foo.go",
		},
		{
			name:     "worktree root file",
			input:    ".worktrees/main/README.md",
			expected: "README.md",
		},
		{
			name:     "worktree without trailing path",
			input:    ".worktrees/branch",
			expected: ".worktrees/branch",
		},
		{
			name:     "vendor path unchanged",
			input:    "vendor/github.com/foo/bar/baz.go",
			expected: "vendor/github.com/foo/bar/baz.go",
		},
		{
			name:     "empty path",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := canonicalPath(tt.input)
			if result != tt.expected {
				t.Errorf("canonicalPath(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestIsWorktreePath(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"pkg/search/search.go", false},
		{".worktrees/branch/pkg/foo.go", true},
		{"some/.worktrees/branch/file.go", true},
		{"worktrees/not-hidden/file.go", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := isWorktreePath(tt.path)
			if result != tt.expected {
				t.Errorf("isWorktreePath(%q) = %v, want %v", tt.path, result, tt.expected)
			}
		})
	}
}

func TestDeduplicateResults_Worktrees(t *testing.T) {
	results := []Result{
		{FilePath: "pkg/core/signature.go", StartLine: 1, EndLine: 10, Score: 0.3},
		{FilePath: ".worktrees/feature/pkg/core/signature.go", StartLine: 1, EndLine: 10, Score: 0.3},
		{FilePath: ".worktrees/another/pkg/core/signature.go", StartLine: 1, EndLine: 10, Score: 0.4},
		{FilePath: "pkg/other/file.go", StartLine: 1, EndLine: 10, Score: 0.5},
	}

	deduped := deduplicateResults(results)

	if len(deduped) != 2 {
		t.Errorf("expected 2 deduplicated results, got %d", len(deduped))
	}

	// Check that we kept the non-worktree path for signature.go
	var foundSignature bool
	for _, r := range deduped {
		if r.FilePath == "pkg/core/signature.go" {
			foundSignature = true
		}
		if isWorktreePath(r.FilePath) && canonicalPath(r.FilePath) == "pkg/core/signature.go" {
			t.Errorf("should prefer non-worktree path, but got %s", r.FilePath)
		}
	}

	if !foundSignature {
		t.Error("expected to keep pkg/core/signature.go as canonical result")
	}
}

func TestDeduplicateResults_WorktreesBetterScore(t *testing.T) {
	// If worktree has better score, it should be kept
	results := []Result{
		{FilePath: "pkg/core/signature.go", StartLine: 1, EndLine: 10, Score: 0.5},
		{FilePath: ".worktrees/feature/pkg/core/signature.go", StartLine: 1, EndLine: 10, Score: 0.2}, // Better score
	}

	deduped := deduplicateResults(results)

	if len(deduped) != 1 {
		t.Errorf("expected 1 deduplicated result, got %d", len(deduped))
	}

	// Should keep the worktree version because it has better score
	if deduped[0].Score != 0.2 {
		t.Errorf("expected score 0.2 (better score wins), got %f", deduped[0].Score)
	}
}

func TestDeduplicateResults_PreferNonWorktreeOnTie(t *testing.T) {
	// On equal scores, prefer non-worktree path
	results := []Result{
		{FilePath: ".worktrees/feature/pkg/foo.go", StartLine: 1, EndLine: 10, Score: 0.3},
		{FilePath: "pkg/foo.go", StartLine: 1, EndLine: 10, Score: 0.3}, // Same score, but comes second
	}

	deduped := deduplicateResults(results)

	if len(deduped) != 1 {
		t.Errorf("expected 1 deduplicated result, got %d", len(deduped))
	}

	// Should prefer the non-worktree path
	if deduped[0].FilePath != "pkg/foo.go" {
		t.Errorf("expected pkg/foo.go (non-worktree preferred on tie), got %s", deduped[0].FilePath)
	}
}

func TestChunksOverlap(t *testing.T) {
	tests := []struct {
		name                       string
		start1, end1, start2, end2 int
		expected                   bool
	}{
		{"no overlap", 1, 10, 20, 30, false},
		{"same range", 1, 10, 1, 10, true},
		{"significant overlap", 1, 10, 5, 15, true},
		{"minor overlap", 1, 100, 99, 200, false}, // Only 2 lines overlap out of 100
		{"contained", 1, 100, 40, 60, true},
		{"adjacent", 1, 10, 11, 20, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := chunksOverlap(tt.start1, tt.end1, tt.start2, tt.end2)
			if result != tt.expected {
				t.Errorf("chunksOverlap(%d-%d, %d-%d) = %v, want %v",
					tt.start1, tt.end1, tt.start2, tt.end2, result, tt.expected)
			}
		})
	}
}
