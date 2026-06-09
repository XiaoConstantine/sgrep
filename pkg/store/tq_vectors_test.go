package store

import (
	"context"
	"math"
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

func randomTQVector(rng *rand.Rand, dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}
	return vec
}
