package store

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestOpenInMem(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()
	if s.db == nil || s.dims != 768 || s.partitions < 2 {
		t.Error("init failed")
	}
}

func TestOpenInMem_CreatesDir(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a", "b", "test.db")
	s, err := OpenInMem(p)
	if err != nil {
		t.Fatal(err)
	}
	_ = s.Close()
}

func TestOpenInMem_CustomDims(t *testing.T) {
	t.Setenv("SGREP_DIMS", "384")
	s := newStore(t)
	defer func() { _ = s.Close() }()
	if s.dims != 384 {
		t.Errorf("got %d", s.dims)
	}
}

func TestStore_Single(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	doc := &Document{
		ID: "d1", FilePath: "/f.go", Content: "x",
		StartLine: 1, EndLine: 2, Embedding: rndVec(768),
		Metadata: map[string]string{"k": "v"},
	}
	if err := s.Store(context.Background(), doc); err != nil {
		t.Fatal(err)
	}
	if len(s.vectors) != 1 || s.docIDs[0] != "d1" {
		t.Error("not stored")
	}
}

func TestStoreBatch(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	docs := make([]*Document, 50)
	for i := range docs {
		docs[i] = &Document{
			ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: rndVec(768),
		}
	}
	if err := s.StoreBatch(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	if len(s.vectors) != 50 {
		t.Errorf("got %d", len(s.vectors))
	}
}

func TestStoreBatch_Empty(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()
	if err := s.StoreBatch(context.Background(), nil); err != nil {
		t.Error(err)
	}
}

func TestSearch_Empty(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()
	docs, dists, err := s.Search(context.Background(), rndVec(768), 10, 5.0)
	if err != nil || len(docs) != 0 || len(dists) != 0 {
		t.Error("should be empty")
	}
}

func TestSearch_Sequential(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	base := rndVec(768)
	docs := make([]*Document, 100)
	for i := range docs {
		e := make([]float32, 768)
		for j := range e {
			e[j] = base[j] + float32(i)*0.01
		}
		docs[i] = &Document{ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: e}
	}
	_ = s.StoreBatch(context.Background(), docs)

	results, dists, _ := s.Search(context.Background(), base, 5, 5.0)
	if len(results) == 0 {
		t.Error("no results")
	}
	for i := 1; i < len(dists); i++ {
		if dists[i] < dists[i-1] {
			t.Error("not sorted")
		}
	}
}

func TestSearch_Parallel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping parallel test in short mode")
	}
	s := newStore(t)
	defer func() { _ = s.Close() }()

	base := rndVec(768)
	docs := make([]*Document, 1200)
	for i := range docs {
		e := make([]float32, 768)
		for j := range e {
			e[j] = base[j] + float32(i)*0.001
		}
		docs[i] = &Document{ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: e}
	}
	_ = s.StoreBatch(context.Background(), docs)

	results, dists, _ := s.Search(context.Background(), base, 10, 50.0)
	if len(results) < 5 {
		t.Errorf("got %d results", len(results))
	}
	for i := 1; i < len(dists); i++ {
		if dists[i] < dists[i-1] {
			t.Error("not sorted")
		}
	}
}

func TestSearch_Threshold(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	near := make([]float32, 768)
	near[0] = 1.0
	far := make([]float32, 768)
	far[0] = 100.0

	_ = s.Store(context.Background(), &Document{ID: "near", FilePath: "/f", Content: "x", Embedding: near})
	_ = s.Store(context.Background(), &Document{ID: "far", FilePath: "/f", Content: "x", Embedding: far})

	query := make([]float32, 768)
	results, _, _ := s.Search(context.Background(), query, 10, 5.0)
	if len(results) != 1 || results[0].ID != "near" {
		t.Error("threshold not working")
	}
}

func TestStats(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	for i := 0; i < 5; i++ {
		_ = s.Store(context.Background(), &Document{
			ID: itoa(i), FilePath: "/f" + itoa(i%2) + ".go", Content: "x", Embedding: rndVec(768),
		})
	}
	stats, _ := s.Stats(context.Background())
	if stats.Chunks != 5 {
		t.Errorf("got %d", stats.Chunks)
	}
}

func TestDeleteByPath(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()

	base := rndVec(768)
	_ = s.Store(context.Background(), &Document{ID: "d1", FilePath: "/a.go", Content: "x", Embedding: base})
	_ = s.Store(context.Background(), &Document{ID: "d2", FilePath: "/a.go", Content: "x", Embedding: base})
	_ = s.Store(context.Background(), &Document{ID: "d3", FilePath: "/b.go", Content: "x", Embedding: base})

	if err := s.DeleteByPath(context.Background(), "/a.go"); err != nil {
		t.Fatal(err)
	}

	// Search should only find /b.go now
	results, _, _ := s.Search(context.Background(), base, 10, 10.0)
	for _, r := range results {
		if r.FilePath == "/a.go" {
			t.Error("deleted docs should not appear")
		}
	}
}

func TestDeleteByPath_NonExistent(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()
	if err := s.DeleteByPath(context.Background(), "/x.go"); err != nil {
		t.Error(err)
	}
}

func TestDeserializeFloat32(t *testing.T) {
	cases := []struct {
		blob []byte
		want []float32
	}{
		{nil, []float32{}},
		{[]byte{}, []float32{}},
		{[]byte{0, 0, 128, 63}, []float32{1.0}},
		{[]byte{0, 0, 128, 63, 0, 0, 0, 64}, []float32{1.0, 2.0}},
		{[]byte{1, 2, 3}, nil}, // invalid
	}
	for _, c := range cases {
		got := deserializeFloat32(c.blob)
		if c.want == nil {
			if got != nil {
				t.Errorf("expected nil for %v", c.blob)
			}
			continue
		}
		if len(got) != len(c.want) {
			t.Errorf("len mismatch: %d vs %d", len(got), len(c.want))
			continue
		}
		for i := range got {
			if math.Abs(float64(got[i]-c.want[i])) > 1e-6 {
				t.Errorf("value[%d] = %f, want %f", i, got[i], c.want[i])
			}
		}
	}
}

func TestConcurrentRead(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping concurrent test in short mode")
	}
	s := newStore(t)
	defer func() { _ = s.Close() }()

	docs := make([]*Document, 100)
	for i := range docs {
		docs[i] = &Document{ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: rndVec(768)}
	}
	_ = s.StoreBatch(context.Background(), docs)

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 3; j++ {
				_, _, _ = s.Search(context.Background(), rndVec(768), 5, 10.0)
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentWrite(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping concurrent test in short mode")
	}
	s := newStore(t)
	defer func() { _ = s.Close() }()

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			docs := make([]*Document, 10)
			for j := range docs {
				docs[j] = &Document{ID: itoa(idx*100 + j), FilePath: "/f.go", Content: "x", Embedding: rndVec(768)}
			}
			_ = s.StoreBatch(context.Background(), docs)
		}(i)
	}
	wg.Wait()

	stats, _ := s.Stats(context.Background())
	if stats.Chunks != 30 {
		t.Errorf("got %d", stats.Chunks)
	}
}

func TestLoadDocuments_Empty(t *testing.T) {
	s := newStore(t)
	defer func() { _ = s.Close() }()
	docs, dists, err := s.loadDocuments(context.Background(), nil)
	if err != nil || docs != nil || dists != nil {
		t.Error("should be nil")
	}
}

func TestClose(t *testing.T) {
	s := newStore(t)
	if err := s.Close(); err != nil {
		t.Error(err)
	}
}

// Benchmarks

func BenchmarkSearch_100(b *testing.B) {
	s := benchStore(b, 100)
	defer func() { _ = s.Close() }()
	q := rndVec(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = s.Search(context.Background(), q, 10, 5.0)
	}
}

func BenchmarkSearch_1000(b *testing.B) {
	s := benchStore(b, 1000)
	defer func() { _ = s.Close() }()
	q := rndVec(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = s.Search(context.Background(), q, 10, 5.0)
	}
}

func BenchmarkStoreBatch_100(b *testing.B) {
	docs := make([]*Document, 100)
	for i := range docs {
		docs[i] = &Document{ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: rndVec(768)}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := benchStoreEmpty(b)
		_ = s.StoreBatch(context.Background(), docs)
		_ = s.Close()
	}
}

// Helpers

func newStore(t *testing.T) *InMemStore {
	s, err := OpenInMem(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func benchStore(b *testing.B, n int) *InMemStore {
	s, _ := OpenInMem(filepath.Join(b.TempDir(), "test.db"))
	docs := make([]*Document, n)
	for i := range docs {
		docs[i] = &Document{ID: itoa(i), FilePath: "/f.go", Content: "x", Embedding: rndVec(768)}
	}
	_ = s.StoreBatch(context.Background(), docs)
	return s
}

func benchStoreEmpty(b *testing.B) *InMemStore {
	s, _ := OpenInMem(filepath.Join(b.TempDir(), "test.db"))
	return s
}

func rndVec(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func itoa(n int) string {
	if n < 0 {
		return "-" + itoa(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}

func init() {
	_ = os.Setenv("SGREP_DIMS", "")
}
