//go:build sqlite_vec
// +build sqlite_vec

package bench

import (
	"context"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/chunk"
	"github.com/XiaoConstantine/sgrep/pkg/store"
)

const searchBenchDims = 768

// BenchmarkSearchEndToEnd benchmarks realistic search scenarios.
func BenchmarkSearchEndToEnd(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping end-to-end search benchmark in short mode")
	}

	queries := []string{
		"How does authentication work?",
		"manage database connections",
		"error handling patterns",
		"parse configuration files",
		"coordinate async operations",
	}

	b.Run("Search_1000docs_SingleQuery", func(b *testing.B) {
		s := setupSearchStore(b, 1000)
		defer func() { _ = s.Close() }()

		query := randomQueryVec(searchBenchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_5000docs_SingleQuery", func(b *testing.B) {
		s := setupSearchStore(b, 5000)
		defer func() { _ = s.Close() }()

		query := randomQueryVec(searchBenchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_10000docs_SingleQuery", func(b *testing.B) {
		s := setupSearchStore(b, 10000)
		defer func() { _ = s.Close() }()

		query := randomQueryVec(searchBenchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = s.Search(context.Background(), query, 10, 5.0)
		}
	})

	b.Run("Search_10000docs_MultiQuery", func(b *testing.B) {
		s := setupSearchStore(b, 10000)
		defer func() { _ = s.Close() }()

		// Pre-generate query vectors for each text query
		queryVecs := make([][]float32, len(queries))
		for i := range queries {
			queryVecs[i] = randomQueryVec(searchBenchDims)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q := queryVecs[i%len(queryVecs)]
			_, _, _ = s.Search(context.Background(), q, 10, 5.0)
		}
	})

	b.Run("Search_10000docs_Concurrent", func(b *testing.B) {
		s := setupSearchStore(b, 10000)
		defer func() { _ = s.Close() }()

		b.RunParallel(func(pb *testing.PB) {
			query := randomQueryVec(searchBenchDims)
			for pb.Next() {
				_, _, _ = s.Search(context.Background(), query, 10, 5.0)
			}
		})
	})

	b.Run("Search_TopK_Comparison", func(b *testing.B) {
		s := setupSearchStore(b, 10000)
		defer func() { _ = s.Close() }()

		query := randomQueryVec(searchBenchDims)

		for _, k := range []int{5, 10, 20, 50} {
			b.Run("TopK_"+intToStr(k), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _, _ = s.Search(context.Background(), query, k, 5.0)
				}
			})
		}
	})
}

// BenchmarkIndexing benchmarks document indexing performance.
func BenchmarkIndexing(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping indexing benchmark in short mode")
	}
	b.Run("StoreBatch_50docs", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, err := store.OpenInMem(dbPath)
		if err != nil {
			b.Fatalf("failed to create store: %v", err)
		}
		defer func() { _ = s.Close() }()

		docs := generateDocs(50)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = s.StoreBatch(context.Background(), docs)
		}
	})

	b.Run("StoreBatch_200docs", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, err := store.OpenInMem(dbPath)
		if err != nil {
			b.Fatalf("failed to create store: %v", err)
		}
		defer func() { _ = s.Close() }()

		docs := generateDocs(200)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = s.StoreBatch(context.Background(), docs)
		}
	})

	b.Run("ChunkAndStore_LargeFile", func(b *testing.B) {
		tmpDir := b.TempDir()
		dbPath := filepath.Join(tmpDir, "bench.db")

		s, err := store.OpenInMem(dbPath)
		if err != nil {
			b.Fatalf("failed to create store: %v", err)
		}
		defer func() { _ = s.Close() }()

		// Generate a realistic Go file
		goCode := generateRealisticGoCode(100)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			chunks, _ := chunk.ChunkFile("/test/main.go", goCode, nil)
			docs := make([]*store.Document, len(chunks))
			for j, c := range chunks {
				docs[j] = &store.Document{
					ID:        "chunk" + intToStr(i*1000+j),
					FilePath:  c.FilePath,
					Content:   c.Content,
					StartLine: c.StartLine,
					EndLine:   c.EndLine,
					Embedding: randomQueryVec(searchBenchDims),
				}
			}
			_ = s.StoreBatch(context.Background(), docs)
		}
	})
}

// BenchmarkSearchWithFiltering benchmarks search with file path filtering.
func BenchmarkSearchWithFiltering(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping search filtering benchmark in short mode")
	}
	b.Run("Search_WithPathFilter", func(b *testing.B) {
		s := setupSearchStoreWithPaths(b, 10000)
		defer func() { _ = s.Close() }()

		query := randomQueryVec(searchBenchDims)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Search returns all, then filter (current behavior)
			results, _, _ := s.Search(context.Background(), query, 50, 5.0)
			filtered := make([]*store.Document, 0, 10)
			for _, r := range results {
				if strings.HasPrefix(r.FilePath, "/pkg/") {
					filtered = append(filtered, r)
					if len(filtered) >= 10 {
						break
					}
				}
			}
		}
	})
}

func setupSearchStore(b *testing.B, numDocs int) *store.InMemStore {
	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench.db")

	s, err := store.OpenInMem(dbPath)
	if err != nil {
		b.Fatalf("failed to create store: %v", err)
	}

	docs := generateDocs(numDocs)
	if err := s.StoreBatch(context.Background(), docs); err != nil {
		b.Fatalf("failed to store docs: %v", err)
	}

	return s
}

func setupSearchStoreWithPaths(b *testing.B, numDocs int) *store.InMemStore {
	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench.db")

	s, err := store.OpenInMem(dbPath)
	if err != nil {
		b.Fatalf("failed to create store: %v", err)
	}

	paths := []string{"/pkg/", "/internal/", "/cmd/", "/test/", "/bench/"}
	docs := make([]*store.Document, numDocs)
	for i := range docs {
		pathPrefix := paths[i%len(paths)]
		docs[i] = &store.Document{
			ID:        "doc" + intToStr(i),
			FilePath:  pathPrefix + "file" + intToStr(i%100) + ".go",
			Content:   "func Test" + intToStr(i) + "() {}",
			StartLine: i,
			EndLine:   i + 10,
			Embedding: randomQueryVec(searchBenchDims),
		}
	}

	if err := s.StoreBatch(context.Background(), docs); err != nil {
		b.Fatalf("failed to store docs: %v", err)
	}

	return s
}

func generateDocs(n int) []*store.Document {
	docs := make([]*store.Document, n)
	for i := range docs {
		docs[i] = &store.Document{
			ID:        "doc" + intToStr(i),
			FilePath:  "/test/file" + intToStr(i%100) + ".go",
			Content:   "func Test" + intToStr(i) + "() {}",
			StartLine: i,
			EndLine:   i + 10,
			Embedding: randomQueryVec(searchBenchDims),
		}
	}
	return docs
}

func randomQueryVec(dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
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

func generateRealisticGoCode(numFuncs int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\n")
	sb.WriteString("import (\n\t\"context\"\n\t\"fmt\"\n\t\"errors\"\n)\n\n")

	for i := 0; i < numFuncs; i++ {
		sb.WriteString("// Handle" + intToStr(i) + " processes request " + intToStr(i) + ".\n")
		sb.WriteString("func Handle" + intToStr(i) + "(ctx context.Context, req *Request) (*Response, error) {\n")
		sb.WriteString("\tif req == nil {\n")
		sb.WriteString("\t\treturn nil, errors.New(\"nil request\")\n")
		sb.WriteString("\t}\n")
		sb.WriteString("\tfmt.Printf(\"handling request %d\\n\", " + intToStr(i) + ")\n")
		sb.WriteString("\treturn &Response{ID: " + intToStr(i) + "}, nil\n")
		sb.WriteString("}\n\n")
	}

	return sb.String()
}
