//go:build !sqlite_vec
// +build !sqlite_vec

package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestLibSQLDiskSize tests disk usage of LibSQLStore.
// Run with: go test -tags=libsql -v -run TestLibSQLDiskSize ./pkg/store/
func TestLibSQLDiskSize(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping disk size test in short mode")
	}

	vectorCounts := []int{100, 500, 1000}
	dims := 768

	for _, count := range vectorCounts {
		t.Run(fmt.Sprintf("vectors_%d", count), func(t *testing.T) {
			docs := generateTestDocs(count, dims)

			dir := t.TempDir()
			dbPath := filepath.Join(dir, "libsql.db")

			s, err := OpenLibSQL(dbPath, WithLibSQLQuantization(QuantizeInt8))
			if err != nil {
				t.Fatalf("OpenLibSQL failed: %v", err)
			}

			ctx := context.Background()
			if err := s.StoreBatch(ctx, docs); err != nil {
				t.Fatalf("StoreBatch failed: %v", err)
				_ = s.Close()
			}

			_ = s.Close()

			size := getDirSize(dir)

			t.Logf("\n=== LibSQL Disk Size (%d vectors, %d dims) ===", count, dims)
			t.Logf("Size: %s (%.2f KB/vector)", formatBytes(size), float64(size)/float64(count)/1024.0)
		})
	}
}

func generateTestDocs(count, dims int) []*Document {
	docs := make([]*Document, count)
	for i := 0; i < count; i++ {
		docs[i] = &Document{
			ID:        fmt.Sprintf("doc-%d", i),
			FilePath:  fmt.Sprintf("file%d.go", i%10),
			Content:   fmt.Sprintf("func test%d() { /* content for function %d */ }", i, i),
			StartLine: i * 10,
			EndLine:   i*10 + 5,
			Embedding: makeTestEmbedding(dims, float32(i)/float32(count)),
			Metadata:  map[string]string{"type": "function", "index": fmt.Sprintf("%d", i)},
		}
	}
	return docs
}

func getDirSize(dir string) int64 {
	var size int64
	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
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
