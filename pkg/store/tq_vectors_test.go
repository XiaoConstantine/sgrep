package store

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

func TestTQVectorStoreSearchRoundTrip(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	rng := rand.New(rand.NewSource(42))

	ids := []string{"chunk_a", "chunk_b", "chunk_c"}
	vectors := make([][]float32, len(ids))
	for i := range vectors {
		vectors[i] = util.NormalizeVector(randomTQVector(rng, dims))
	}

	count, err := BuildTQVectorStore(ctx, tmp, ids, vectors, TQVectorBuildOptions{
		Dims: dims,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}
	if count != len(ids) {
		t.Fatalf("count = %d, want %d", count, len(ids))
	}

	store, err := OpenTQVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQVectorStore: %v", err)
	}
	defer func() { _ = store.Close() }()

	if store.VectorCount() != len(ids) {
		t.Fatalf("VectorCount = %d, want %d", store.VectorCount(), len(ids))
	}
	if store.Dims() != dims {
		t.Fatalf("Dims = %d, want %d", store.Dims(), dims)
	}
	if store.Bits() != 4 {
		t.Fatalf("Bits = %d, want 4", store.Bits())
	}

	results, err := store.Search(ctx, vectors[1], 2, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}
	if results[0].ID != "chunk_b" {
		t.Fatalf("top result = %q, want chunk_b; results=%v", results[0].ID, results)
	}
	if math.IsNaN(results[0].Distance) || math.IsInf(results[0].Distance, 0) {
		t.Fatalf("invalid distance: %v", results[0].Distance)
	}
}

func TestTQVectorAccumulatorWritesSearchableChunkStore(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64

	acc, err := NewTQVectorAccumulator(TQVectorBuildOptions{Dims: dims, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatalf("NewTQVectorAccumulator: %v", err)
	}
	query := make([]float32, dims)
	query[0] = 1
	other := make([]float32, dims)
	other[1] = 1
	if err := acc.Add("query", query); err != nil {
		t.Fatalf("Add query: %v", err)
	}
	if err := acc.Add("other", other); err != nil {
		t.Fatalf("Add other: %v", err)
	}
	if acc.Count() != 2 {
		t.Fatalf("Count = %d, want 2", acc.Count())
	}
	count, err := acc.WriteChunkStore(ctx, tmp)
	if err != nil {
		t.Fatalf("WriteChunkStore: %v", err)
	}
	if count != 2 {
		t.Fatalf("written count = %d, want 2", count)
	}

	store, err := OpenTQVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQVectorStore: %v", err)
	}
	defer func() { _ = store.Close() }()
	results, err := store.Search(ctx, query, 1, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 || results[0].ID != "query" {
		t.Fatalf("results = %+v, want query", results)
	}
}

func TestTQFileVectorStoreSearchRoundTrip(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	rng := rand.New(rand.NewSource(43))

	paths := []string{"README.md", "internal/auth.go", "pkg/store.go"}
	vectors := make([][]float32, len(paths))
	for i := range vectors {
		vectors[i] = util.NormalizeVector(randomTQVector(rng, dims))
	}
	count, err := BuildTQFileVectorStore(ctx, tmp, paths, vectors, TQVectorBuildOptions{
		Dims: dims,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		t.Fatalf("BuildTQFileVectorStore: %v", err)
	}
	if count != len(paths) {
		t.Fatalf("count = %d, want %d", count, len(paths))
	}

	store, err := OpenTQFileVectorStore(tmp)
	if err != nil {
		t.Fatalf("OpenTQFileVectorStore: %v", err)
	}
	defer func() { _ = store.Close() }()

	results, err := store.Search(ctx, vectors[1], 1, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 || results[0].ID != "internal/auth.go" {
		t.Fatalf("file result = %+v, want internal/auth.go", results)
	}
}

func TestTQSearchStoreHydratesInDenseOrder(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	rng := rand.New(rand.NewSource(7))

	ids := []string{"chunk_a", "chunk_b", "chunk_c"}
	vectors := make([][]float32, len(ids))
	for i := range vectors {
		vectors[i] = util.NormalizeVector(randomTQVector(rng, dims))
	}
	if _, err := BuildTQVectorStore(ctx, tmp, ids, vectors, TQVectorBuildOptions{Dims: dims, Bits: 4, Seed: 42}); err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}

	base := &stubHydrationStore{
		docs: map[string]*Document{
			"chunk_a": {ID: "chunk_a", FilePath: "a.go"},
			"chunk_b": {ID: "chunk_b", FilePath: "b.go"},
			"chunk_c": {ID: "chunk_c", FilePath: "c.go"},
		},
	}
	wrapped, err := OpenTQSearchStoreIfAvailable(base, tmp)
	if err != nil {
		t.Fatalf("OpenTQSearchStoreIfAvailable: %v", err)
	}
	defer func() { _ = wrapped.Close() }()

	docs, _, err := wrapped.Search(ctx, vectors[2], 3, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("Search returned no docs")
	}
	if docs[0].ID != "chunk_c" {
		t.Fatalf("first doc = %q, want chunk_c", docs[0].ID)
	}
}

func TestSearchWrapperPrefersExactArtifactWhenAvailable(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	query := make([]float32, dims)
	query[0] = 1
	if _, err := BuildTQVectorStore(ctx, tmp, []string{"doc"}, [][]float32{query}, TQVectorBuildOptions{Dims: dims}); err != nil {
		t.Fatal(err)
	}
	exact, err := OpenMMapVectorStore(tmp, dims)
	if err != nil {
		t.Fatal(err)
	}
	exact.BeginWrite()
	exact.WriteVector("doc", query)
	if err := exact.CommitWrite(); err != nil {
		t.Fatal(err)
	}
	_ = exact.Close()

	base := &stubHydrationStore{docs: map[string]*Document{"doc": {ID: "doc", FilePath: "doc.go"}}}
	wrapped, err := OpenTQSearchStoreIfAvailable(base, tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = wrapped.Close() }()
	searchStore := wrapped.(*TQSearchStore)
	if _, ok := searchStore.dense.(*MMapVectorStore); !ok {
		t.Fatalf("dense artifact type = %T, want exact mmap", searchStore.dense)
	}
}

func TestTQSearchStoreHybridFallsBackToSemanticOnBM25Error(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64
	rng := rand.New(rand.NewSource(11))

	ids := []string{"chunk_a", "chunk_b"}
	vectors := make([][]float32, len(ids))
	for i := range vectors {
		vectors[i] = util.NormalizeVector(randomTQVector(rng, dims))
	}
	if _, err := BuildTQVectorStore(ctx, tmp, ids, vectors, TQVectorBuildOptions{Dims: dims, Bits: 4, Seed: 42}); err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}

	base := &stubBM25ErrorStore{
		stubHydrationStore: stubHydrationStore{
			docs: map[string]*Document{
				"chunk_a": {ID: "chunk_a", FilePath: "a.go"},
				"chunk_b": {ID: "chunk_b", FilePath: "b.go"},
			},
		},
	}
	wrapped, err := OpenTQSearchStoreIfAvailable(base, tmp)
	if err != nil {
		t.Fatalf("OpenTQSearchStoreIfAvailable: %v", err)
	}
	defer func() { _ = wrapped.Close() }()

	docs, _, err := wrapped.HybridSearch(ctx, vectors[0], `"unterminated`, 1, 2, 0.6, 0.4)
	if err != nil {
		t.Fatalf("HybridSearch returned BM25 error instead of semantic fallback: %v", err)
	}
	if len(docs) == 0 || docs[0].ID != "chunk_a" {
		t.Fatalf("HybridSearch fallback docs = %v, want chunk_a first", docs)
	}
}

func TestTQSearchStoreHybridUnionsDenseAndBM25Candidates(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	dims := 64

	query := make([]float32, dims)
	query[0] = 1
	lexicalVec := make([]float32, dims)
	lexicalVec[1] = 1
	if _, err := BuildTQVectorStore(ctx, tmp,
		[]string{"dense", "lexical"},
		[][]float32{query, lexicalVec},
		TQVectorBuildOptions{Dims: dims, Bits: 4, Seed: 42},
	); err != nil {
		t.Fatalf("BuildTQVectorStore: %v", err)
	}

	base := &stubBM25SearchStore{
		stubHydrationStore: stubHydrationStore{
			docs: map[string]*Document{
				"dense":   {ID: "dense", FilePath: "dense.go"},
				"lexical": {ID: "lexical", FilePath: "lexical.go"},
			},
		},
		results: []BM25SearchResult{{ID: "lexical", Score: -1}},
	}
	wrapped, err := OpenTQSearchStoreIfAvailable(base, tmp)
	if err != nil {
		t.Fatalf("OpenTQSearchStoreIfAvailable: %v", err)
	}
	defer func() { _ = wrapped.Close() }()

	docs, _, err := wrapped.HybridSearch(ctx, query, "lexical", 1, 0.2, 0.1, 0.9)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "lexical" {
		t.Fatalf("HybridSearch docs = %v, want lexical BM25 candidate first", docs)
	}

	docs, _, err = wrapped.HybridSearch(ctx, query, "lexical", 1, -1, 0.1, 0.9)
	if err != nil {
		t.Fatalf("HybridSearch with no dense hits: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "lexical" {
		t.Fatalf("HybridSearch with no dense hits docs = %v, want lexical BM25 candidate", docs)
	}
}

type stubHydrationStore struct {
	docs map[string]*Document
}

func (s *stubHydrationStore) Store(context.Context, *Document) error        { return nil }
func (s *stubHydrationStore) StoreBatch(context.Context, []*Document) error { return nil }
func (s *stubHydrationStore) Search(context.Context, []float32, int, float64) ([]*Document, []float64, error) {
	return nil, nil, nil
}
func (s *stubHydrationStore) HybridSearch(context.Context, []float32, string, int, float64, float64, float64) ([]*Document, []float64, error) {
	return nil, nil, nil
}
func (s *stubHydrationStore) Stats(context.Context) (*Stats, error)      { return &Stats{}, nil }
func (s *stubHydrationStore) DeleteByPath(context.Context, string) error { return nil }
func (s *stubHydrationStore) Close() error                               { return nil }

func (s *stubHydrationStore) LoadDocumentsByID(_ context.Context, ids []string) (map[string]*Document, error) {
	out := make(map[string]*Document, len(ids))
	for _, id := range ids {
		if doc, ok := s.docs[id]; ok {
			out[id] = doc
		}
	}
	return out, nil
}

type stubBM25ErrorStore struct {
	stubHydrationStore
}

func (s *stubBM25ErrorStore) BM25Scores(context.Context, string) (map[string]float64, error) {
	return nil, errors.New("fts syntax error")
}

type stubBM25SearchStore struct {
	stubHydrationStore
	results []BM25SearchResult
}

func (s *stubBM25SearchStore) BM25Scores(context.Context, string) (map[string]float64, error) {
	scores := make(map[string]float64, len(s.results))
	for _, result := range s.results {
		scores[result.ID] = result.Score
	}
	return scores, nil
}

func (s *stubBM25SearchStore) BM25Search(_ context.Context, _ string, limit int) ([]BM25SearchResult, error) {
	results := append([]BM25SearchResult(nil), s.results...)
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func BenchmarkVectorBackendScale(b *testing.B) {
	for _, count := range []int{10_000, 100_000} {
		b.Run(fmt.Sprintf("vectors=%d", count), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			vectors := make([][]float32, count)
			binaryVectors := make([][]byte, count)
			ids := make([]string, count)
			for i := range vectors {
				vectors[i] = util.NormalizeVector(randomTQVector(rng, 768))
				binaryVectors[i] = QuantizeToBinary(vectors[i])
				ids[i] = fmt.Sprintf("doc-%08d", i)
			}
			query := vectors[0]
			queryBinary := binaryVectors[0]
			distances := make([]float64, count)

			b.Run("exact-float32-scan", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(count * 768 * 4))
				for i := 0; i < b.N; i++ {
					util.DotProductDistanceBatch(query, vectors, distances)
				}
			})

			b.Run("binary-coarse-scan", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(count * len(queryBinary)))
				for i := 0; i < b.N; i++ {
					total := 0
					for _, code := range binaryVectors {
						for j, value := range code {
							total += bits.OnesCount8(value ^ queryBinary[j])
						}
					}
					if total < 0 {
						b.Fatal(total)
					}
				}
			})

			dir := b.TempDir()
			exactStore, err := OpenMMapVectorStore(dir, 768)
			if err != nil {
				b.Fatal(err)
			}
			exactStore.BeginWrite()
			for i, id := range ids {
				exactStore.WriteVector(id, vectors[i])
			}
			if err := exactStore.CommitWrite(); err != nil {
				b.Fatal(err)
			}
			b.Run("exact-float32-top50", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(count * 768 * 4))
				for i := 0; i < b.N; i++ {
					if _, err := exactStore.Search(context.Background(), query, 50, 2); err != nil {
						b.Fatal(err)
					}
				}
			})
			_ = exactStore.Close()

			if _, err := BuildTQVectorStore(context.Background(), dir, ids, vectors, TQVectorBuildOptions{Dims: 768, Bits: 4, Seed: 42}); err != nil {
				b.Fatal(err)
			}
			tq, err := OpenTQVectorStore(dir)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = tq.Close() }()
			b.Run("tqmse-top50", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(count * tq.codeSize))
				for i := 0; i < b.N; i++ {
					if _, err := tq.Search(context.Background(), query, 50, 2); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func randomTQVector(rng *rand.Rand, dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}
	return vec
}
