//go:build ignore

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/XiaoConstantine/sgrep/pkg/store"
)

func main() {
	counts := []int{100, 500, 1000, 5000}
	dims := 768

	fmt.Println("=== Store Disk Size Comparison ===")
	fmt.Println()

	for _, count := range counts {
		docs := generateDocs(count, dims)

		// Test sqlite-vec (BufferedStore)
		sqliteSize := testBufferedStore(docs)

		// Test libsql would require different build
		// For now, just show sqlite-vec results

		fmt.Printf("Vectors: %d (dims=%d)\n", count, dims)
		fmt.Printf("  sqlite-vec: %s (%.2f KB/vector)\n",
			formatBytes(sqliteSize), float64(sqliteSize)/float64(count)/1024.0)
		fmt.Println()
	}
}

func generateDocs(count, dims int) []*store.Document {
	docs := make([]*store.Document, count)
	for i := 0; i < count; i++ {
		embedding := make([]float32, dims)
		for j := range embedding {
			embedding[j] = float32(i*dims+j) / float32(count*dims)
		}
		docs[i] = &store.Document{
			ID:        fmt.Sprintf("doc-%d", i),
			FilePath:  fmt.Sprintf("file%d.go", i%100),
			Content:   fmt.Sprintf("func test%d() { /* content */ }", i),
			StartLine: i * 10,
			EndLine:   i*10 + 5,
			Embedding: embedding,
		}
	}
	return docs
}

func testBufferedStore(docs []*store.Document) int64 {
	dir, _ := os.MkdirTemp("", "sgrep-bench-*")
	defer os.RemoveAll(dir)

	dbPath := filepath.Join(dir, "test.db")
	s, err := store.OpenBuffered(dbPath, store.WithBufferedQuantization(store.QuantizeInt8))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return 0
	}

	ctx := context.Background()
	if err := s.StoreBatch(ctx, docs); err != nil {
		fmt.Printf("StoreBatch error: %v\n", err)
		_ = s.Close()
		return 0
	}

	if err := s.Flush(ctx); err != nil {
		fmt.Printf("Flush error: %v\n", err)
	}

	_ = s.Close()

	return getDirSize(dir)
}

func getDirSize(dir string) int64 {
	var size int64
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}
