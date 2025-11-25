package bench

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/internal/chunk"
	"github.com/XiaoConstantine/sgrep/internal/store"
	"github.com/XiaoConstantine/sgrep/internal/util"
)

const (
	benchDims = 768
)

func BenchmarkVectorOperations(b *testing.B) {
	b.Run("L2Distance_768dims", func(b *testing.B) {
		a := randomVec(benchDims)
		b2 := randomVec(benchDims)
		scratch := make([]float32, benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.L2DistanceSlab(a, b2, scratch)
		}
	})

	b.Run("L2DistanceBatch_1000_768dims", func(b *testing.B) {
		query := randomVec(benchDims)
		vectors := make([][]float32, 1000)
		for i := range vectors {
			vectors[i] = randomVec(benchDims)
		}
		distances := make([]float64, 1000)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.L2DistanceBatch(query, vectors, distances)
		}
	})

	b.Run("L2DistanceBatch_10000_768dims", func(b *testing.B) {
		query := randomVec(benchDims)
		vectors := make([][]float32, 10000)
		for i := range vectors {
			vectors[i] = randomVec(benchDims)
		}
		distances := make([]float64, 10000)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.L2DistanceBatch(query, vectors, distances)
		}
	})

	b.Run("TopKIndices_10000_K10", func(b *testing.B) {
		distances := make([]float64, 10000)
		indices := make([]int, 10000)
		for i := range distances {
			distances[i] = rand.Float64()
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.TopKIndices(distances, indices, 10)
		}
	})

	b.Run("TopKIndices_10000_K100", func(b *testing.B) {
		distances := make([]float64, 10000)
		indices := make([]int, 10000)
		for i := range distances {
			distances[i] = rand.Float64()
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.TopKIndices(distances, indices, 100)
		}
	})
}

func BenchmarkSlabPool(b *testing.B) {
	b.Run("Get_16partitions", func(b *testing.B) {
		pool := util.NewSlabPool(16, 10000, benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pool.Get(i % 16)
		}
	})

	b.Run("SlabReuse", func(b *testing.B) {
		pool := util.NewSlabPool(4, 5000, benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			slab := pool.Get(i % 4)
			distances := slab.Distances(1000)
			for j := range distances {
				distances[j] = float64(j)
			}
		}
	})
}

func BenchmarkQueryCache(b *testing.B) {
	b.Run("Set", func(b *testing.B) {
		cache := util.NewQueryCache(10000, 5*time.Minute)
		value := "cached value"

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.Set("key", value)
		}
	})

	b.Run("Get_Hit", func(b *testing.B) {
		cache := util.NewQueryCache(10000, 5*time.Minute)
		cache.Set("key", "value")

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.Get("key")
		}
	})

	b.Run("Get_Miss", func(b *testing.B) {
		cache := util.NewQueryCache(10000, 5*time.Minute)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.Get("nonexistent")
		}
	})

	b.Run("Concurrent_ReadWrite", func(b *testing.B) {
		cache := util.NewQueryCache(10000, 5*time.Minute)

		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					cache.Set("key", i)
				} else {
					cache.Get("key")
				}
				i++
			}
		})
	})
}

func BenchmarkEventBox(b *testing.B) {
	b.Run("Set", func(b *testing.B) {
		eb := util.NewEventBox()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eb.Set(util.EvtSearchProgress, i)
		}
	})

	b.Run("Peek", func(b *testing.B) {
		eb := util.NewEventBox()
		eb.Set(util.EvtSearchProgress, 42)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eb.Peek(util.EvtSearchProgress)
		}
	})

	b.Run("Concurrent", func(b *testing.B) {
		eb := util.NewEventBox()

		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				if i%2 == 0 {
					eb.Set(util.EvtSearchProgress, i)
				} else {
					eb.Peek(util.EvtSearchProgress)
				}
				i++
			}
		})
	})
}

func BenchmarkInMemStore(b *testing.B) {
	setupStore := func(b *testing.B, numDocs int) *store.InMemStore {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, err := store.OpenInMem(dbPath)
		if err != nil {
			b.Fatalf("failed to create store: %v", err)
		}

		docs := make([]*store.Document, numDocs)
		for i := range docs {
			docs[i] = &store.Document{
				ID:        "doc" + intToStr(i),
				FilePath:  "/test/file" + intToStr(i%100) + ".go",
				Content:   "func Test" + intToStr(i) + "() {}",
				StartLine: i,
				EndLine:   i + 10,
				Embedding: randomVec(benchDims),
			}
		}

		if err := s.StoreBatch(context.Background(), docs); err != nil {
			b.Fatalf("failed to store docs: %v", err)
		}

		return s
	}

	b.Run("Search_100docs", func(b *testing.B) {
		s := setupStore(b, 100)
		defer func() { _ = s.Close() }()

		query := randomVec(benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_1000docs", func(b *testing.B) {
		s := setupStore(b, 1000)
		defer func() { _ = s.Close() }()

		query := randomVec(benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_10000docs", func(b *testing.B) {
		s := setupStore(b, 10000)
		defer func() { _ = s.Close() }()

		query := randomVec(benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_50000docs", func(b *testing.B) {
		if testing.Short() {
			b.Skip("skipping large benchmark in short mode")
		}

		s := setupStore(b, 50000)
		defer func() { _ = s.Close() }()

		query := randomVec(benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Store_Single", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, _ := store.OpenInMem(dbPath)
		defer func() { _ = s.Close() }()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			doc := &store.Document{
				ID:        "doc" + intToStr(i),
				FilePath:  "/test.go",
				Content:   "content",
				Embedding: randomVec(benchDims),
			}
			_ = s.Store(context.Background(), doc)
		}
	})

	b.Run("StoreBatch_100", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, _ := store.OpenInMem(dbPath)
		defer func() { _ = s.Close() }()

		docs := make([]*store.Document, 100)
		for i := range docs {
			docs[i] = &store.Document{
				ID:        "doc" + intToStr(i),
				FilePath:  "/test.go",
				Content:   "content",
				Embedding: randomVec(benchDims),
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = s.StoreBatch(context.Background(), docs)
		}
	})

	b.Run("Concurrent_Search_10000docs", func(b *testing.B) {
		s := setupStore(b, 10000)
		defer func() { _ = s.Close() }()

		b.RunParallel(func(pb *testing.PB) {
			query := randomVec(benchDims)
			for pb.Next() {
				_, _, _ = s.Search(context.Background(), query, 10, 5.0)
			}
		})
	})
}

func BenchmarkChunking(b *testing.B) {
	goContent := generateGoCode(100)
	textContent := strings.Repeat("This is a line of text content.\n", 200)

	b.Run("ChunkFile_Go_100funcs", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = chunk.ChunkFile("/test/main.go", goContent, nil)
		}
	})

	b.Run("ChunkFile_Text_200lines", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = chunk.ChunkFile("/test/file.txt", textContent, nil)
		}
	})

	b.Run("ChunkFile_Large_1000lines", func(b *testing.B) {
		largeContent := generateGoCode(500)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = chunk.ChunkFile("/test/large.go", largeContent, nil)
		}
	})
}

func BenchmarkEndToEnd(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping end-to-end benchmark in short mode")
	}

	b.Run("Index_And_Search", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, _ := store.OpenInMem(dbPath)
		defer func() { _ = s.Close() }()

		goContent := generateGoCode(50)
		chunks, _ := chunk.ChunkFile("/test/main.go", goContent, nil)

		docs := make([]*store.Document, len(chunks))
		for i, c := range chunks {
			docs[i] = &store.Document{
				ID:        "chunk" + intToStr(i),
				FilePath:  c.FilePath,
				Content:   c.Content,
				StartLine: c.StartLine,
				EndLine:   c.EndLine,
				Embedding: randomVec(benchDims),
			}
		}
		_ = s.StoreBatch(context.Background(), docs)

		query := randomVec(benchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})
}

func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("VectorSlab_NoAlloc", func(b *testing.B) {
		slab := util.NewVectorSlab(1000, benchDims)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			distances := slab.Distances(500)
			_ = distances
		}
	})

	b.Run("SlabPool_GetReuse", func(b *testing.B) {
		pool := util.NewSlabPool(8, 1000, benchDims)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			slab := pool.Get(i % 8)
			distances := slab.Distances(100)
			_ = distances
		}
	})

	b.Run("L2DistanceBatch_PreAlloc", func(b *testing.B) {
		query := randomVec(benchDims)
		vectors := make([][]float32, 100)
		for i := range vectors {
			vectors[i] = randomVec(benchDims)
		}
		distances := make([]float64, 100)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.L2DistanceBatch(query, vectors, distances)
		}
	})
}

func randomVec(dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

func generateGoCode(numFuncs int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nimport \"fmt\"\n\n")

	for i := 0; i < numFuncs; i++ {
		sb.WriteString("// Function" + intToStr(i) + " does something.\n")
		sb.WriteString("func Function" + intToStr(i) + "(x int) int {\n")
		sb.WriteString("    result := x * 2\n")
		sb.WriteString("    fmt.Println(result)\n")
		sb.WriteString("    return result\n")
		sb.WriteString("}\n\n")
	}

	return sb.String()
}

func intToStr(n int) string {
	if n < 0 {
		return "-" + intToStr(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return intToStr(n/10) + string(rune('0'+n%10))
}

func TestBenchmarkSetup(t *testing.T) {
	if os.Getenv("BENCHMARK_SETUP_TEST") == "" {
		t.Skip("skipping benchmark setup test")
	}
}
