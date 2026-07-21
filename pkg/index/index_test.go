package index

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	searchpkg "github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

type segmentCountTestStore struct {
	chunks []store.ChunkInfo
}

func (s *segmentCountTestStore) StoreColBERTSegments(ctx context.Context, chunkID string, segments []store.ColBERTSegment) error {
	return nil
}

func (s *segmentCountTestStore) StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]store.ColBERTSegment) error {
	return nil
}

func (s *segmentCountTestStore) GetColBERTSegments(ctx context.Context, chunkID string) ([]store.ColBERTSegment, error) {
	return nil, nil
}

func (s *segmentCountTestStore) GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]store.ColBERTSegment, error) {
	return map[string][]store.ColBERTSegment{}, nil
}

func (s *segmentCountTestStore) DeleteColBERTSegments(ctx context.Context, chunkID string) error {
	return nil
}

func (s *segmentCountTestStore) HasColBERTSegments(ctx context.Context) (bool, error) {
	return false, nil
}

func (s *segmentCountTestStore) GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]store.ChunkInfo, error) {
	if offset >= len(s.chunks) {
		return nil, nil
	}
	end := offset + batchSize
	if end > len(s.chunks) {
		end = len(s.chunks)
	}
	return append([]store.ChunkInfo(nil), s.chunks[offset:end]...), nil
}

type exportColBERTTestStore struct {
	chunks   []store.ChunkInfo
	segments map[string][]store.ColBERTSegment
	codec    store.ColBERTCodec
	pq       *util.ProductQuantizer
	tq       *util.TQMSEQuantizer
}

func (s *exportColBERTTestStore) Store(ctx context.Context, doc *store.Document) error {
	return nil
}

func (s *exportColBERTTestStore) StoreBatch(ctx context.Context, docs []*store.Document) error {
	return nil
}

func (s *exportColBERTTestStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*store.Document, []float64, error) {
	return nil, nil, nil
}

func (s *exportColBERTTestStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*store.Document, []float64, error) {
	return nil, nil, nil
}

func (s *exportColBERTTestStore) Stats(ctx context.Context) (*store.Stats, error) {
	return &store.Stats{Chunks: int64(len(s.chunks))}, nil
}

func (s *exportColBERTTestStore) DeleteByPath(ctx context.Context, filepath string) error {
	return nil
}

func (s *exportColBERTTestStore) Close() error {
	return nil
}

func (s *exportColBERTTestStore) StoreColBERTSegments(ctx context.Context, chunkID string, segments []store.ColBERTSegment) error {
	if s.segments == nil {
		s.segments = make(map[string][]store.ColBERTSegment)
	}
	s.segments[chunkID] = append([]store.ColBERTSegment(nil), segments...)
	return nil
}

func (s *exportColBERTTestStore) StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]store.ColBERTSegment) error {
	if s.segments == nil {
		s.segments = make(map[string][]store.ColBERTSegment)
	}
	for chunkID, segments := range chunkSegments {
		s.segments[chunkID] = append([]store.ColBERTSegment(nil), segments...)
	}
	return nil
}

func (s *exportColBERTTestStore) GetColBERTSegments(ctx context.Context, chunkID string) ([]store.ColBERTSegment, error) {
	return append([]store.ColBERTSegment(nil), s.segments[chunkID]...), nil
}

func (s *exportColBERTTestStore) GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]store.ColBERTSegment, error) {
	result := make(map[string][]store.ColBERTSegment, len(chunkIDs))
	for _, chunkID := range chunkIDs {
		if segments, ok := s.segments[chunkID]; ok {
			result[chunkID] = append([]store.ColBERTSegment(nil), segments...)
		}
	}
	return result, nil
}

func (s *exportColBERTTestStore) DeleteColBERTSegments(ctx context.Context, chunkID string) error {
	delete(s.segments, chunkID)
	return nil
}

func (s *exportColBERTTestStore) HasColBERTSegments(ctx context.Context) (bool, error) {
	return len(s.segments) > 0, nil
}

func (s *exportColBERTTestStore) GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]store.ChunkInfo, error) {
	if offset >= len(s.chunks) {
		return nil, nil
	}
	end := offset + batchSize
	if end > len(s.chunks) {
		end = len(s.chunks)
	}
	return append([]store.ChunkInfo(nil), s.chunks[offset:end]...), nil
}

func (s *exportColBERTTestStore) ColBERTCodec() store.ColBERTCodec {
	return s.codec
}

func (s *exportColBERTTestStore) ProductQuantizer() *util.ProductQuantizer {
	return s.pq
}

func (s *exportColBERTTestStore) TQMSEQuantizer() *util.TQMSEQuantizer {
	return s.tq
}

type tqExportTestStore struct {
	ids          []string
	vectors      [][]float32
	filePaths    []string
	fileVectors  [][]float32
	checkpointed bool
}

func (s *tqExportTestStore) Store(context.Context, *store.Document) error {
	return nil
}

func (s *tqExportTestStore) StoreBatch(context.Context, []*store.Document) error {
	return nil
}

func (s *tqExportTestStore) Search(context.Context, []float32, int, float64) ([]*store.Document, []float64, error) {
	return nil, nil, nil
}

func (s *tqExportTestStore) HybridSearch(context.Context, []float32, string, int, float64, float64, float64) ([]*store.Document, []float64, error) {
	return nil, nil, nil
}

func (s *tqExportTestStore) Stats(context.Context) (*store.Stats, error) {
	return &store.Stats{Chunks: int64(len(s.ids))}, nil
}

func (s *tqExportTestStore) DeleteByPath(context.Context, string) error {
	return nil
}

func (s *tqExportTestStore) Close() error {
	return nil
}

func (s *tqExportTestStore) ExportAllVectors(context.Context) ([]string, [][]float32, error) {
	return append([]string(nil), s.ids...), append([][]float32(nil), s.vectors...), nil
}

func (s *tqExportTestStore) ExportFileEmbeddings(context.Context) ([]string, [][]float32, error) {
	return append([]string(nil), s.filePaths...), append([][]float32(nil), s.fileVectors...), nil
}

func (s *tqExportTestStore) Checkpoint(context.Context) error {
	s.checkpointed = true
	return nil
}

type vectorClearTestStore struct {
	tqExportTestStore
	cleared bool
}

func (s *vectorClearTestStore) ClearVectorStorage(context.Context) error {
	s.cleared = true
	return nil
}

type pruneIndexTestStore struct {
	tqExportTestStore
	liveIDs   []string
	livePaths []string
}

func (s *pruneIndexTestStore) PruneIndex(ctx context.Context, liveIDs []string, livePaths []string) error {
	s.liveIDs = append([]string(nil), liveIDs...)
	s.livePaths = append([]string(nil), livePaths...)
	return nil
}

func TestGetSgrepHome(t *testing.T) {
	t.Run("from_env", func(t *testing.T) {
		t.Setenv("SGREP_HOME", "/custom/path")
		home, err := getSgrepHome()
		if err != nil {
			t.Fatal(err)
		}
		if home != "/custom/path" {
			t.Errorf("got %s, want /custom/path", home)
		}
	})

	t.Run("default", func(t *testing.T) {
		t.Setenv("SGREP_HOME", "")
		home, err := getSgrepHome()
		if err != nil {
			t.Fatal(err)
		}
		homeDir, _ := os.UserHomeDir()
		if home != filepath.Join(homeDir, ".sgrep") {
			t.Errorf("got %s", home)
		}
	})
}

func TestHashPath(t *testing.T) {
	h1 := hashPath("/path/one")
	h2 := hashPath("/path/two")
	if len(h1) != 12 || h1 == h2 {
		t.Errorf("hash issue: %s vs %s", h1, h2)
	}
	if hashPath("/path/one") != h1 {
		t.Error("not deterministic")
	}
}

func TestWriteRepoMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := writeRepoMetadata(dir, "/test/repo"); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if len(data) == 0 {
		t.Error("empty metadata")
	}
	if err := ValidateRepoMetadata(dir); err != nil {
		t.Fatalf("ValidateRepoMetadata: %v", err)
	}
}

func TestValidateRepoMetadataRejectsLegacyIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(`{"path":"/legacy"}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ValidateRepoMetadata(dir); err == nil || !strings.Contains(err.Error(), "run 'sgrep index .'") {
		t.Fatalf("ValidateRepoMetadata legacy error = %v", err)
	}
}

func TestIsCodeFile(t *testing.T) {
	cases := map[string]bool{
		"main.go": true, "app.ts": true, "x.py": true, "x.rs": true,
		"x.java": true, "x.c": true, "x.cpp": true, "x.rb": true,
		"x.md": true, "x.json": true, "x.yaml": true, "x.toml": true,
		"x.png": false, "x.exe": false, "noext": false,
	}
	for path, want := range cases {
		if got := isCodeFile(path); got != want {
			t.Errorf("isCodeFile(%q)=%v want %v", path, got, want)
		}
	}
}

func TestIsKnownIgnoreDir(t *testing.T) {
	for _, d := range []string{"node_modules", "vendor", ".git", "dist", "build"} {
		if !isKnownIgnoreDir(d) {
			t.Errorf("%s should be ignored", d)
		}
	}
	if isKnownIgnoreDir("src") {
		t.Error("src should not be ignored")
	}
}

func TestIgnoreRules(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("*.log\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, ".sgrepignore"), []byte("custom/\n"), 0644)

	ir := NewIgnoreRules(dir)

	tests := []struct {
		path string
		want bool
	}{
		{dir, false},
		{filepath.Join(dir, "node_modules"), true},
		{filepath.Join(dir, ".git"), true},
		{filepath.Join(dir, "src"), false},
		{filepath.Join(dir, "app.log"), true},
		{filepath.Join(dir, "app.min.js"), true},
	}
	for _, tt := range tests {
		if got := ir.ShouldIgnore(tt.path); got != tt.want {
			t.Errorf("ShouldIgnore(%q)=%v want %v", tt.path, got, tt.want)
		}
	}
}

func TestIgnoreRules_LoadMissing(t *testing.T) {
	ir := &IgnoreRules{rootPath: "/nonexistent"}
	ir.ensureRulesLoaded("/nonexistent")
}

func TestIndexer_Fields(t *testing.T) {
	idx := &Indexer{rootPath: "/test", processed: 5, errors: 2}
	if idx.rootPath != "/test" || idx.processed != 5 || idx.errors != 2 {
		t.Error("field issue")
	}
}

func TestPruneStaleIndexPassesSortedLiveSets(t *testing.T) {
	fake := &pruneIndexTestStore{}
	idx := &Indexer{store: fake}

	pruned, err := idx.pruneStaleIndex(context.Background(),
		map[string]struct{}{"b.go:chunk_1": {}, "a.go:chunk_1": {}},
		map[string]struct{}{"b.go": {}, "a.go": {}},
	)
	if err != nil {
		t.Fatalf("pruneStaleIndex failed: %v", err)
	}
	if !pruned {
		t.Fatal("expected pruneStaleIndex to use store pruner")
	}
	if want := []string{"a.go:chunk_1", "b.go:chunk_1"}; !reflect.DeepEqual(fake.liveIDs, want) {
		t.Fatalf("live IDs = %v, want %v", fake.liveIDs, want)
	}
	if want := []string{"a.go", "b.go"}; !reflect.DeepEqual(fake.livePaths, want) {
		t.Fatalf("live paths = %v, want %v", fake.livePaths, want)
	}
}

func TestRebuildTQVectorStoreRemovesStaleArtifactWhenNoVectors(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64

	if _, err := store.BuildTQVectorStore(ctx, tmp, []string{"stale"}, [][]float32{{1}}, store.TQVectorBuildOptions{Dims: 1}); err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}
	if !store.HasTQVectorStore(tmp) {
		t.Fatal("expected stale TQ vector store to exist")
	}
	if _, err := store.BuildTQFileVectorStore(ctx, tmp, []string{"stale.go"}, [][]float32{{1}}, store.TQVectorBuildOptions{Dims: 1}); err != nil {
		t.Fatalf("BuildTQFileVectorStore: %v", err)
	}
	if !store.HasTQFileVectorStore(tmp) {
		t.Fatal("expected stale file TQ vector store to exist")
	}

	fake := &tqExportTestStore{}
	idx := &Indexer{repoDir: tmp, store: fake}
	t.Setenv("SGREP_DIMS", fmt.Sprintf("%d", dims))

	count, err := idx.RebuildTQVectorStore(ctx)
	if err != nil {
		t.Fatalf("RebuildTQVectorStore: %v", err)
	}
	if count != 0 {
		t.Fatalf("count = %d, want 0", count)
	}
	if store.HasTQVectorStore(tmp) {
		t.Fatal("stale TQ vector store was not removed")
	}
	if store.HasTQFileVectorStore(tmp) {
		t.Fatal("stale file TQ vector store was not removed")
	}
	if !fake.checkpointed {
		t.Fatal("expected SQL checkpoint before TQ export")
	}
}

func TestRebuildTQVectorStoreExportsFileVectors(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	chunkVec := make([]float32, dims)
	fileVec := make([]float32, dims)
	chunkVec[0] = 1
	fileVec[1] = 1

	fake := &tqExportTestStore{
		ids:         []string{"a.go:chunk_1"},
		vectors:     [][]float32{chunkVec},
		filePaths:   []string{"a.go"},
		fileVectors: [][]float32{fileVec},
	}
	idx := &Indexer{repoDir: tmp, store: fake}
	t.Setenv("SGREP_DIMS", fmt.Sprintf("%d", dims))

	count, err := idx.RebuildTQVectorStore(ctx)
	if err != nil {
		t.Fatalf("RebuildTQVectorStore: %v", err)
	}
	if count != 1 {
		t.Fatalf("count = %d, want 1", count)
	}
	if !store.HasTQVectorStore(tmp) {
		t.Fatal("expected chunk TQ vector store")
	}
	if !store.HasTQFileVectorStore(tmp) {
		t.Fatal("expected file TQ vector store")
	}
	fileStore, err := store.OpenTQFileVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQFileVectorStore: %v", err)
	}
	defer func() { _ = fileStore.Close() }()
	hits, err := fileStore.Search(ctx, fileVec, 1, 2)
	if err != nil {
		t.Fatalf("Search file TQ vectors: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "a.go" {
		t.Fatalf("file hits = %+v, want a.go", hits)
	}
}

func TestRefreshDerivedVectorArtifactsSkipsCompactExportAfterErrors(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	t.Setenv("SGREP_DIMS", fmt.Sprintf("%d", dims))

	oldChunkVec := make([]float32, dims)
	oldChunkVec[0] = 1
	oldFileVec := make([]float32, dims)
	oldFileVec[0] = 1
	opts := store.TQVectorBuildOptions{
		Dims: dims,
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	}
	if _, err := store.BuildTQVectorStore(ctx, tmp, []string{"old.go:chunk_1"}, [][]float32{oldChunkVec}, opts); err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}
	if _, err := store.BuildTQFileVectorStore(ctx, tmp, []string{"old.go"}, [][]float32{oldFileVec}, opts); err != nil {
		t.Fatalf("BuildTQFileVectorStore: %v", err)
	}

	partialVec := make([]float32, dims)
	partialVec[1] = 1
	collector, err := newCompactVectorCollector()
	if err != nil {
		t.Fatalf("newCompactVectorCollector: %v", err)
	}
	if err := collector.AddDocuments([]*store.Document{
		{ID: "partial.go:chunk_1", FilePath: "partial.go", Embedding: partialVec, StartLine: 1, EndLine: 10},
	}); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	fake := &vectorClearTestStore{}
	idx := &Indexer{repoDir: tmp, store: fake, compactTQ: collector}
	if err := idx.refreshDerivedVectorArtifacts(ctx, util.NewTimingStats(util.DebugOff), true); err != nil {
		t.Fatalf("refreshDerivedVectorArtifacts: %v", err)
	}
	if idx.tqBuilt {
		t.Fatal("compact TQ export marked built after indexing errors")
	}
	if fake.cleared {
		t.Fatal("SQL vectors were cleared after indexing errors")
	}

	chunkStore, err := store.OpenTQVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQVectorStore: %v", err)
	}
	defer func() { _ = chunkStore.Close() }()
	if got := chunkStore.VectorCount(); got != 1 {
		t.Fatalf("chunk vector count = %d, want old store count 1", got)
	}
	chunkScores, err := chunkStore.ScoreByID(ctx, oldChunkVec, []string{"old.go:chunk_1", "partial.go:chunk_1"})
	if err != nil {
		t.Fatalf("ScoreByID chunk store: %v", err)
	}
	if _, ok := chunkScores["old.go:chunk_1"]; !ok {
		t.Fatalf("old chunk vector missing after skipped export: %v", chunkScores)
	}
	if _, ok := chunkScores["partial.go:chunk_1"]; ok {
		t.Fatalf("partial chunk vector was exported after indexing errors: %v", chunkScores)
	}

	fileStore, err := store.OpenTQFileVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQFileVectorStore: %v", err)
	}
	defer func() { _ = fileStore.Close() }()
	if got := fileStore.VectorCount(); got != 1 {
		t.Fatalf("file vector count = %d, want old store count 1", got)
	}
	fileScores, err := fileStore.ScoreByID(ctx, oldFileVec, []string{"old.go", "partial.go"})
	if err != nil {
		t.Fatalf("ScoreByID file store: %v", err)
	}
	if _, ok := fileScores["old.go"]; !ok {
		t.Fatalf("old file vector missing after skipped export: %v", fileScores)
	}
	if _, ok := fileScores["partial.go"]; ok {
		t.Fatalf("partial file vector was exported after indexing errors: %v", fileScores)
	}
}

func TestCompactVectorCollectorWritesChunkAndFileStores(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	t.Setenv("SGREP_DIMS", fmt.Sprintf("%d", dims))

	a1 := make([]float32, dims)
	a1[0] = 1
	a2 := make([]float32, dims)
	a2[0] = 0.9
	a2[1] = 0.1
	a2 = util.NormalizeVector(a2)
	b1 := make([]float32, dims)
	b1[1] = 1
	collector, err := newCompactVectorCollector()
	if err != nil {
		t.Fatalf("newCompactVectorCollector: %v", err)
	}
	if err := collector.AddDocuments([]*store.Document{
		{ID: "a.go:chunk_1", FilePath: "a.go", Embedding: a1, StartLine: 1, EndLine: 10},
		{ID: "a.go:chunk_2", FilePath: "a.go", Embedding: a2, StartLine: 11, EndLine: 20},
		{ID: "b.go:chunk_1", FilePath: "b.go", Embedding: b1, StartLine: 1, EndLine: 10},
	}); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	chunkCount, fileCount, err := collector.Write(ctx, tmp)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if chunkCount != 3 {
		t.Fatalf("chunk count = %d, want 3", chunkCount)
	}
	if fileCount != 2 {
		t.Fatalf("file count = %d, want 2", fileCount)
	}

	chunkStore, err := store.OpenTQVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQVectorStore: %v", err)
	}
	defer func() { _ = chunkStore.Close() }()
	chunkHits, err := chunkStore.Search(ctx, a1, 1, 2)
	if err != nil {
		t.Fatalf("Search chunk store: %v", err)
	}
	if len(chunkHits) != 1 || chunkHits[0].ID != "a.go:chunk_1" {
		t.Fatalf("chunk hits = %+v, want a.go:chunk_1", chunkHits)
	}

	exactStore, err := store.OpenMMapVectorStore(tmp, dims)
	if err != nil {
		t.Fatalf("OpenMMapVectorStore: %v", err)
	}
	defer func() { _ = exactStore.Close() }()
	exactHits, err := exactStore.Search(ctx, a1, 1, 2)
	if err != nil || len(exactHits) != 1 || exactHits[0].ID != "a.go:chunk_1" {
		t.Fatalf("exact hits = %+v, err=%v", exactHits, err)
	}

	fileStore, err := store.OpenTQFileVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQFileVectorStore: %v", err)
	}
	defer func() { _ = fileStore.Close() }()
	fileQuery := make([]float32, dims)
	for i := range fileQuery {
		fileQuery[i] = (a1[i] + a2[i]) / 2
	}
	fileHits, err := fileStore.Search(ctx, fileQuery, 1, 2)
	if err != nil {
		t.Fatalf("Search file store: %v", err)
	}
	if len(fileHits) != 1 || fileHits[0].ID != "a.go" {
		t.Fatalf("file hits = %+v, want a.go", fileHits)
	}
}

func TestDefaultIndexConfig(t *testing.T) {
	cfg := DefaultIndexConfig()
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if cfg.Workers < 4 {
		t.Errorf("workers should be at least 4, got %d", cfg.Workers)
	}
	if cfg.Workers > 32 {
		t.Errorf("workers should be capped at 32, got %d", cfg.Workers)
	}
	if cfg.EmbedConcurrency < 4 {
		t.Errorf("EmbedConcurrency should be at least 4, got %d", cfg.EmbedConcurrency)
	}
	if cfg.EmbedConcurrency > 16 {
		t.Errorf("EmbedConcurrency should be capped at 16, got %d", cfg.EmbedConcurrency)
	}
}

func TestDecomposeDocumentForColBERT_AdaptiveUsesRawSegments(t *testing.T) {
	content := makeSegmentedContent(16)

	raw := searchpkg.DecomposeDocumentRaw(content)
	adaptive := decomposeDocumentForColBERT(content, true)
	legacy := decomposeDocumentForColBERT(content, false)
	budget := searchpkg.AdaptiveSegmentBudget(content)

	if len(raw) <= budget {
		t.Fatalf("test content did not exceed adaptive budget: raw=%d budget=%d", len(raw), budget)
	}
	if len(adaptive) != len(raw) {
		t.Fatalf("adaptive decomposition should keep raw segments before pooling: got %d want %d", len(adaptive), len(raw))
	}
	if len(legacy) >= len(adaptive) {
		t.Fatalf("legacy decomposition should stay capped below raw adaptive split: legacy=%d adaptive=%d", len(legacy), len(adaptive))
	}
}

func TestBuildStoredColBERTSegments_AdaptivePoolsToBudget(t *testing.T) {
	segmentTexts := make([]string, 16)
	embeddings := make([][]float32, len(segmentTexts))
	budget := searchpkg.AdaptiveSegmentBudgetFromRawCount(len(segmentTexts))

	rng := rand.New(rand.NewSource(42))
	for i := range segmentTexts {
		segmentTexts[i] = fmt.Sprintf("segment %d", i)
		embeddings[i] = randomNormalizedEmbedding(rng, 768)
	}

	segments := buildStoredColBERTSegments("chunk-a", segmentTexts, embeddings, true, store.ColBERTCodecInt8, nil, nil)

	if len(segments) > budget {
		t.Fatalf("adaptive pooling exceeded budget: got %d want <= %d", len(segments), budget)
	}
	if len(segments) >= len(segmentTexts) {
		t.Fatalf("adaptive pooling did not reduce segment count: got %d from %d", len(segments), len(segmentTexts))
	}
	for i, seg := range segments {
		if seg.SegmentIdx != i {
			t.Fatalf("segment %d has idx %d", i, seg.SegmentIdx)
		}
	}
}

func TestBuildStoredColBERTSegments_AdaptiveKeepsLegacySizedChunks(t *testing.T) {
	segmentTexts := make([]string, 6)
	embeddings := make([][]float32, len(segmentTexts))

	rng := rand.New(rand.NewSource(99))
	for i := range segmentTexts {
		segmentTexts[i] = fmt.Sprintf("segment %d", i)
		embeddings[i] = randomNormalizedEmbedding(rng, 768)
	}

	segments := buildStoredColBERTSegments("chunk-b", segmentTexts, embeddings, true, store.ColBERTCodecInt8, nil, nil)
	if len(segments) != len(segmentTexts) {
		t.Fatalf("legacy-sized chunk should remain unchanged: got %d want %d", len(segments), len(segmentTexts))
	}
}

func TestApplyPQCodecSizeGate_DowngradesSmallCorpus(t *testing.T) {
	chunks := make([]store.ChunkInfo, 100)
	for i := range chunks {
		chunks[i] = store.ChunkInfo{
			ID:      fmt.Sprintf("chunk-%d", i),
			Content: makeSegmentedContent(10),
		}
	}

	idx := &Indexer{
		indexCfg:     &IndexConfig{AdaptiveColBERTSegments: false},
		colbertCodec: store.ColBERTCodecPQ6,
	}
	err := idx.applyPQCodecSizeGate(context.Background(), &segmentCountTestStore{chunks: chunks})
	if err != nil {
		t.Fatalf("applyPQCodecSizeGate failed: %v", err)
	}
	if idx.colbertCodec != store.ColBERTCodecInt8 {
		t.Fatalf("expected small corpus to downgrade to int8, got %s", idx.colbertCodec)
	}
	if idx.colbertPQ != nil {
		t.Fatal("expected downgrade to clear PQ codebook")
	}
}

func TestApplyPQCodecSizeGate_KeepsPQForLargeCorpus(t *testing.T) {
	chunks := make([]store.ChunkInfo, 600)
	for i := range chunks {
		chunks[i] = store.ChunkInfo{
			ID:      fmt.Sprintf("chunk-%d", i),
			Content: makeSegmentedContent(10),
		}
	}

	idx := &Indexer{
		indexCfg:     &IndexConfig{AdaptiveColBERTSegments: false},
		colbertCodec: store.ColBERTCodecPQ6,
	}
	err := idx.applyPQCodecSizeGate(context.Background(), &segmentCountTestStore{chunks: chunks})
	if err != nil {
		t.Fatalf("applyPQCodecSizeGate failed: %v", err)
	}
	if idx.colbertCodec != store.ColBERTCodecPQ6 {
		t.Fatalf("expected large corpus to keep pq6, got %s", idx.colbertCodec)
	}
}

func TestApplyPQCodecSizeGate_AdaptiveUsesBudgetedEstimate(t *testing.T) {
	content := makeSegmentedContent(15)
	rawCount := len(searchpkg.DecomposeDocumentRaw(content))
	legacyCount := len(searchpkg.DecomposeDocument(content))
	budget := searchpkg.AdaptiveSegmentBudgetFromRawCount(rawCount)
	if rawCount <= legacyCount {
		t.Fatalf("test content did not exceed legacy decomposition: raw=%d legacy=%d", rawCount, legacyCount)
	}
	if budget >= rawCount {
		t.Fatalf("adaptive budget did not shrink raw count: raw=%d budget=%d", rawCount, budget)
	}

	chunks := make([]store.ChunkInfo, 1000)
	for i := range chunks {
		chunks[i] = store.ChunkInfo{
			ID:      fmt.Sprintf("chunk-%d", i),
			Content: content,
		}
	}

	idx := &Indexer{
		indexCfg:     &IndexConfig{AdaptiveColBERTSegments: true},
		colbertCodec: store.ColBERTCodecPQ6,
	}
	err := idx.applyPQCodecSizeGate(context.Background(), &segmentCountTestStore{chunks: chunks})
	if err != nil {
		t.Fatalf("applyPQCodecSizeGate failed: %v", err)
	}
	if idx.colbertCodec != store.ColBERTCodecInt8 {
		t.Fatalf("expected adaptive budgeted estimate %d to downgrade to int8, got %s", budget*len(chunks), idx.colbertCodec)
	}
}

func TestBuildStoredColBERTSegments_AdaptiveMergesSimilarSegments(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	baseA := randomNormalizedEmbedding(rng, 768)
	baseB := randomNormalizedEmbedding(rng, 768)
	baseC := randomNormalizedEmbedding(rng, 768)
	baseD := randomNormalizedEmbedding(rng, 768)

	segmentTexts := []string{"a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2", "d0", "d1", "d2"}
	embeddings := [][]float32{
		perturbEmbedding(baseA, 0.002),
		perturbEmbedding(baseA, 0.003),
		perturbEmbedding(baseA, 0.004),
		perturbEmbedding(baseB, 0.002),
		perturbEmbedding(baseB, 0.003),
		perturbEmbedding(baseB, 0.004),
		perturbEmbedding(baseC, 0.002),
		perturbEmbedding(baseC, 0.003),
		perturbEmbedding(baseC, 0.004),
		perturbEmbedding(baseD, 0.002),
		perturbEmbedding(baseD, 0.003),
		perturbEmbedding(baseD, 0.004),
	}

	segments := buildStoredColBERTSegments("chunk-c", segmentTexts, embeddings, true, store.ColBERTCodecInt8, nil, nil)
	if len(segments) >= len(segmentTexts) {
		t.Fatalf("adaptive merge did not collapse similar segments: got %d from %d", len(segments), len(segmentTexts))
	}
	if len(segments) > searchpkg.AdaptiveSegmentBudgetFromRawCount(len(segmentTexts)) {
		t.Fatalf("expected merged result to respect adaptive budget, got %d", len(segments))
	}
	for i, seg := range segments {
		if len(seg.EmbeddingInt8) == 0 {
			t.Fatalf("merged segment %d lost its int8 payload", i)
		}
	}
}

func TestBuildFloat32ColBERTSegments_AdaptiveMergesKeepFloatEmbeddings(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	baseA := randomNormalizedEmbedding(rng, 768)
	baseB := randomNormalizedEmbedding(rng, 768)

	segmentTexts := []string{"a0", "a1", "a2", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"}
	embeddings := [][]float32{
		perturbEmbedding(baseA, 0.002),
		perturbEmbedding(baseA, 0.003),
		perturbEmbedding(baseA, 0.004),
		perturbEmbedding(baseB, 0.002),
		perturbEmbedding(baseB, 0.003),
		perturbEmbedding(baseB, 0.004),
		perturbEmbedding(baseB, 0.005),
		perturbEmbedding(baseB, 0.006),
		perturbEmbedding(baseB, 0.007),
		perturbEmbedding(baseB, 0.008),
		perturbEmbedding(baseB, 0.009),
		perturbEmbedding(baseB, 0.010),
	}

	segments := buildFloat32ColBERTSegments(segmentTexts, embeddings, true)
	if len(segments) >= len(segmentTexts) {
		t.Fatalf("adaptive float32 merge did not reduce segment count: got %d from %d", len(segments), len(segmentTexts))
	}
	for i, seg := range segments {
		if len(seg.Embedding) == 0 {
			t.Fatalf("float32 merged segment %d lost its embedding", i)
		}
		if len(seg.EmbeddingInt8) != 0 {
			t.Fatalf("float32 merged segment %d should not be quantized yet", i)
		}
	}
}

func TestBuildStoredColBERTSegments_PQ6EncodesCodes(t *testing.T) {
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
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}, 4); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	segments := buildStoredColBERTSegments(
		"chunk-pq",
		[]string{"a", "b"},
		[][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
		},
		false,
		store.ColBERTCodecPQ6,
		pq,
		nil,
	)

	if len(segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(segments))
	}
	if len(segments[0].PQCodes) != pq.CodeSize() {
		t.Fatalf("expected %d PQ bytes, got %d", pq.CodeSize(), len(segments[0].PQCodes))
	}
	if len(segments[0].EmbeddingInt8) != 0 {
		t.Fatalf("expected PQ path to omit int8 payload")
	}
}

func TestBuildStoredColBERTSegments_TQMSEEncodesCodes(t *testing.T) {
	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: 4,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}

	segments := buildStoredColBERTSegments(
		"chunk-tq",
		[]string{"a", "b"},
		[][]float32{
			util.NormalizeVectorCopy([]float32{1, 0, 0, 0}),
			util.NormalizeVectorCopy([]float32{0, 1, 0, 0}),
		},
		false,
		store.ColBERTCodecTQMSE,
		nil,
		tq,
	)

	if len(segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(segments))
	}
	if len(segments[0].TQCodes) != tq.CodeSize() {
		t.Fatalf("expected %d TQ bytes, got %d", tq.CodeSize(), len(segments[0].TQCodes))
	}
	if len(segments[0].EmbeddingInt8) != 0 || len(segments[0].PQCodes) != 0 {
		t.Fatalf("expected TQ path to omit int8/PQ payloads")
	}
}

func TestExportColBERTToMMap_ReencodesLegacyInt8SegmentsToTQMSE(t *testing.T) {
	t.Setenv("SGREP_DIMS", "4")
	ctx := context.Background()

	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: 4,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}

	emb := util.NormalizeVectorCopy([]float32{0.92, 0.21, -0.26, 0.11})
	quantized, scale, min := util.QuantizeInt8(emb)
	legacySegments := []store.ColBERTSegment{
		{
			SegmentIdx:    0,
			Text:          "legacy int8",
			EmbeddingInt8: quantized,
			QuantScale:    scale,
			QuantMin:      min,
		},
	}
	expected, err := encodeSegmentsToTQMSE("chunk-old", legacySegments, tq)
	if err != nil {
		t.Fatalf("encodeSegmentsToTQMSE failed: %v", err)
	}
	if allBytesZero(expected[0].TQCodes) {
		t.Fatal("test fixture produced an all-zero TQ-MSE code")
	}

	idx := &Indexer{
		store: &exportColBERTTestStore{
			chunks: []store.ChunkInfo{{ID: "chunk-old"}},
			segments: map[string][]store.ColBERTSegment{
				"chunk-old": legacySegments,
			},
			codec: store.ColBERTCodecTQMSE,
			tq:    tq,
		},
	}

	outputDir := t.TempDir()
	total, err := idx.ExportColBERTToMMap(ctx, outputDir)
	if err != nil {
		t.Fatalf("ExportColBERTToMMap failed: %v", err)
	}
	if total != 1 {
		t.Fatalf("expected 1 exported segment, got %d", total)
	}

	mmapStore, err := store.OpenMMapSegmentStore(outputDir, 4)
	if err != nil {
		t.Fatalf("OpenMMapSegmentStore failed: %v", err)
	}
	defer func() { _ = mmapStore.Close() }()

	if mmapStore.ColBERTCodec() != store.ColBERTCodecTQMSE {
		t.Fatalf("expected mmap codec %q, got %q", store.ColBERTCodecTQMSE, mmapStore.ColBERTCodec())
	}
	segments, err := mmapStore.GetColBERTSegments(ctx, "chunk-old")
	if err != nil {
		t.Fatalf("GetColBERTSegments failed: %v", err)
	}
	if len(segments) != 1 {
		t.Fatalf("expected 1 mmap segment, got %d", len(segments))
	}
	if !bytes.Equal(segments[0].TQCodes, expected[0].TQCodes) {
		t.Fatalf("expected exported TQ code %v, got %v", expected[0].TQCodes, segments[0].TQCodes)
	}
	if len(segments[0].EmbeddingInt8) != 0 || len(segments[0].PQCodes) != 0 {
		t.Fatalf("expected exported segment to contain only TQ codes")
	}
}

func TestEncodeSegmentsToTQMSE_PreservesStoredCodes(t *testing.T) {
	tq, err := util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: 4,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("NewTQMSEQuantizer failed: %v", err)
	}
	code, err := tq.Encode(util.NormalizeVectorCopy([]float32{0.2, 0.8, -0.1, 0.3}))
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	encoded, err := encodeSegmentsToTQMSE("chunk-tq", []store.ColBERTSegment{
		{SegmentIdx: 0, Text: "stored tq", TQCodes: code.Codes},
	}, tq)
	if err != nil {
		t.Fatalf("encodeSegmentsToTQMSE failed: %v", err)
	}
	if len(encoded) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(encoded))
	}
	if !bytes.Equal(encoded[0].TQCodes, code.Codes) {
		t.Fatalf("expected preserved code %v, got %v", code.Codes, encoded[0].TQCodes)
	}
	encoded[0].TQCodes[0] ^= 0xff
	if bytes.Equal(encoded[0].TQCodes, code.Codes) {
		t.Fatal("expected preserved code to be copied, not aliased")
	}
}

func TestEncodeSegmentsToPQ_UsesStoredInt8Embeddings(t *testing.T) {
	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       4,
		Subspaces:  2,
		Centroids:  4,
		Iterations: 4,
	})
	if err != nil {
		t.Fatalf("NewProductQuantizer failed: %v", err)
	}
	training := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	if err := pq.Train(training, 4); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	q0, scale0, min0 := util.QuantizeInt8([]float32{1, 0, 0, 0})
	q1, scale1, min1 := util.QuantizeInt8([]float32{0, 1, 0, 0})
	encoded, err := encodeSegmentsToPQ("chunk-int8", []store.ColBERTSegment{
		{
			SegmentIdx:    0,
			Text:          "a",
			EmbeddingInt8: q0,
			QuantScale:    scale0,
			QuantMin:      min0,
		},
		{
			SegmentIdx:    1,
			Text:          "b",
			EmbeddingInt8: q1,
			QuantScale:    scale1,
			QuantMin:      min1,
		},
	}, pq)
	if err != nil {
		t.Fatalf("encodeSegmentsToPQ failed: %v", err)
	}

	if len(encoded) != 2 {
		t.Fatalf("expected 2 encoded segments, got %d", len(encoded))
	}
	if encoded[0].SegmentIdx != 0 || encoded[1].SegmentIdx != 1 {
		t.Fatalf("segment indexes were not preserved: %+v", encoded)
	}
	if encoded[0].Text != "a" || encoded[1].Text != "b" {
		t.Fatalf("segment texts were not preserved: %+v", encoded)
	}
	if len(encoded[0].PQCodes) != pq.CodeSize() || len(encoded[1].PQCodes) != pq.CodeSize() {
		t.Fatalf("expected code size %d, got %d/%d", pq.CodeSize(), len(encoded[0].PQCodes), len(encoded[1].PQCodes))
	}
	if len(encoded[0].EmbeddingInt8) != 0 || len(encoded[1].EmbeddingInt8) != 0 {
		t.Fatalf("encoded PQ segments should not keep int8 payloads: %+v", encoded)
	}
}

func TestConvertColBERTSegmentsForCodecStrict_PQEncodeFailureErrors(t *testing.T) {
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
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}, 4); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	_, err = convertColBERTSegmentsForCodecStrict("chunk-strict", []store.ColBERTSegment{
		{SegmentIdx: 0, Text: "broken"},
	}, store.ColBERTCodecPQ6, pq, nil)
	if err == nil {
		t.Fatal("expected strict PQ conversion to fail for missing embeddings")
	}
}

func allBytesZero(data []byte) bool {
	for _, b := range data {
		if b != 0 {
			return false
		}
	}
	return len(data) > 0
}

func TestColBERTPQScratchRoundTrip(t *testing.T) {
	writer, err := newColBERTPQScratchWriter(t.TempDir())
	if err != nil {
		t.Fatalf("newColBERTPQScratchWriter failed: %v", err)
	}

	first := []store.ColBERTSegment{
		{SegmentIdx: 0, Text: "first", Embedding: []float32{1, 0, 0, 0}},
		{SegmentIdx: 1, Text: "second", Embedding: []float32{0, 1, 0, 0}},
	}
	second := []store.ColBERTSegment{
		{SegmentIdx: 0, Text: "third", Embedding: []float32{0, 0, 1, 0}},
	}
	if err := writer.WriteChunk("chunk-1", first); err != nil {
		t.Fatalf("WriteChunk failed: %v", err)
	}
	if err := writer.WriteChunk("chunk-2", second); err != nil {
		t.Fatalf("WriteChunk second failed: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	reader, err := openColBERTPQScratchReader(path)
	if err != nil {
		t.Fatalf("openColBERTPQScratchReader failed: %v", err)
	}
	defer func() { _ = reader.Close() }()

	record, err := reader.Next()
	if err != nil {
		t.Fatalf("Next failed: %v", err)
	}
	if record.ChunkID != "chunk-1" {
		t.Fatalf("unexpected chunk id %q", record.ChunkID)
	}
	if len(record.Segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(record.Segments))
	}
	if len(record.Segments[0].Embedding) != 4 || record.Segments[0].Embedding[0] != 1 {
		t.Fatalf("unexpected first embedding: %+v", record.Segments[0].Embedding)
	}

	record, err = reader.Next()
	if err != nil {
		t.Fatalf("second Next failed: %v", err)
	}
	if record.ChunkID != "chunk-2" {
		t.Fatalf("unexpected second chunk id %q", record.ChunkID)
	}
	if len(record.Segments) != 1 || record.Segments[0].Text != "third" {
		t.Fatalf("unexpected second record: %+v", record)
	}

	record, err = reader.Next()
	if err != io.EOF {
		t.Fatalf("expected EOF after scratch stream, got record=%+v err=%v", record, err)
	}
}

func TestIsTestFile(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		// Go test files
		{"main_test.go", true},
		{"pkg/foo_test.go", true},
		{"main.go", false},

		// JS/TS test files
		{"app.test.ts", true},
		{"app.test.tsx", true},
		{"app.test.js", true},
		{"app.test.jsx", true},
		{"app.spec.ts", true},
		{"app.spec.tsx", true},
		{"app.spec.js", true},
		{"app.spec.jsx", true},
		{"app.ts", false},
		{"app.js", false},

		// Python test files
		{"test_main.py", true},
		{"main_test.py", true},
		{"main.py", false},

		// Ruby test files
		{"main_spec.rb", true},
		{"main.rb", false},

		// Rust test files
		{"main_test.rs", true},
		{"main.rs", false},

		// Java test files
		{"MainTest.java", true},
		{"MainTests.java", true},
		{"Main.java", false},

		// Files in test directories (need proper path structure)
		{filepath.Join("src", "tests", "main.go"), true},
		{filepath.Join("src", "test", "main.py"), true},
		{filepath.Join("src", "__tests__", "app.js"), true},
		{filepath.Join("src", "spec", "helper.rb"), true},
		{filepath.Join("src", "specs", "main.rb"), true},
		{filepath.Join("src", "_tests", "foo.go"), true},

		// Non-test files
		{"src/main.go", false},
		{"lib/util.py", false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := isTestFile(tt.path)
			if got != tt.want {
				t.Errorf("isTestFile(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func makeSegmentedContent(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "func segment%d() {\n", i)
		fmt.Fprintf(&b, "    value%d := %d\n", i, i)
		fmt.Fprintf(&b, "    return value%d\n", i)
		b.WriteString("}\n\n")
	}
	return b.String()
}

func randomNormalizedEmbedding(rng *rand.Rand, dims int) []float32 {
	emb := make([]float32, dims)
	for i := range emb {
		emb[i] = rng.Float32()*2 - 1
	}
	return util.NormalizeVector(emb)
}

func perturbEmbedding(base []float32, noise float32) []float32 {
	emb := make([]float32, len(base))
	copy(emb, base)
	for i := range emb {
		delta := noise
		if i%2 == 0 {
			delta = -noise
		}
		emb[i] += delta
	}
	return util.NormalizeVector(emb)
}

func TestIgnoreRules_Patterns(t *testing.T) {
	dir := t.TempDir()

	// Create .gitignore with various patterns
	gitignore := `# Comment
*.log
build/
*.min.js
`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(gitignore), 0644)

	ir := NewIgnoreRules(dir)

	tests := []struct {
		path string
		want bool
	}{
		// Default ignores
		{filepath.Join(dir, "node_modules"), true},
		{filepath.Join(dir, "vendor"), true},
		{filepath.Join(dir, "__pycache__"), true},
		{filepath.Join(dir, ".idea"), true},
		{filepath.Join(dir, ".vscode"), true},
		{filepath.Join(dir, "dist"), true},
		{filepath.Join(dir, "build"), true},
		{filepath.Join(dir, ".git"), true},
		{filepath.Join(dir, ".sgrep"), true},

		// From .gitignore
		{filepath.Join(dir, "app.log"), true},
		{filepath.Join(dir, "bundle.min.js"), true},

		// Should not ignore
		{filepath.Join(dir, "src"), false},
		{filepath.Join(dir, "main.go"), false},
		{filepath.Join(dir, "app.js"), false},
	}

	for _, tt := range tests {
		t.Run(filepath.Base(tt.path), func(t *testing.T) {
			got := ir.ShouldIgnore(tt.path)
			if got != tt.want {
				t.Errorf("ShouldIgnore(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestIgnoreRules_CommentLines(t *testing.T) {
	dir := t.TempDir()

	// .gitignore with comments
	content := `# This is a comment
*.log
# Another comment
`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644)

	ir := NewIgnoreRules(dir)

	// Comments should not be treated as patterns
	if ir.ShouldIgnore(filepath.Join(dir, "# This is a comment")) {
		t.Error("comment line should not be used as pattern")
	}
}

func TestIgnoreRules_EmptyLines(t *testing.T) {
	dir := t.TempDir()

	content := `*.log

*.tmp

`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644)

	ir := NewIgnoreRules(dir)

	// Should ignore .log and .tmp files
	if !ir.ShouldIgnore(filepath.Join(dir, "app.log")) {
		t.Error("should ignore .log files")
	}
	if !ir.ShouldIgnore(filepath.Join(dir, "temp.tmp")) {
		t.Error("should ignore .tmp files")
	}
}

func TestIgnoreRules_NestedGitignore(t *testing.T) {
	dir := t.TempDir()
	genDir := filepath.Join(dir, "pkg", "generated")
	if err := os.MkdirAll(genDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pkg", ".gitignore"), []byte("generated/\n"), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnore(filepath.Join(genDir, "client.go")) {
		t.Fatal("nested .gitignore directory rule should ignore descendants")
	}
	if ir.ShouldIgnore(filepath.Join(dir, "pkg", "service.go")) {
		t.Fatal("nested .gitignore should not ignore siblings outside the rule")
	}
}

func TestIgnoreRules_Negation(t *testing.T) {
	dir := t.TempDir()
	content := "*.go\n!keep.go\n"
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnore(filepath.Join(dir, "drop.go")) {
		t.Fatal("expected generic ignore pattern to match")
	}
	if ir.ShouldIgnore(filepath.Join(dir, "keep.go")) {
		t.Fatal("negated rule should re-include keep.go")
	}
}

func TestIgnoreRules_AnchoredPattern(t *testing.T) {
	dir := t.TempDir()
	content := "/generated/\n"
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnorePath(filepath.Join(dir, "generated"), true) {
		t.Fatal("anchored root directory should be ignored")
	}
	if ir.ShouldIgnorePath(filepath.Join(dir, "pkg", "generated"), true) {
		t.Fatal("anchored pattern should not match nested directory")
	}
}
