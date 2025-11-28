//go:build ignore

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/chunk"
	"github.com/XiaoConstantine/sgrep/pkg/embed"
)

func main() {
	ProfileIndexing()
}

// ProfileIndexing measures time spent in each phase of indexing
func ProfileIndexing() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run index_profile.go <path-to-file>")
		os.Exit(1)
	}

	filePath := os.Args[1]
	ctx := context.Background()

	// Read file
	t0 := time.Now()
	content, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	readTime := time.Since(t0)
	fmt.Printf("1. File read (%d bytes): %v\n", len(content), readTime)

	// Chunk file
	t1 := time.Now()
	cfg := chunk.DefaultConfig()
	chunks, err := chunk.ChunkFile(filepath.Base(filePath), string(content), cfg)
	if err != nil {
		panic(err)
	}
	chunkTime := time.Since(t1)
	fmt.Printf("2. Chunking (%d chunks): %v\n", len(chunks), chunkTime)

	// Create embedder
	t2 := time.Now()
	embedder := embed.New()
	embedderInitTime := time.Since(t2)
	fmt.Printf("3. Embedder init: %v\n", embedderInitTime)

	// Prepare all texts
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		if c.Description != "" {
			texts[i] = c.Description + "\n\n" + c.Content
		} else {
			texts[i] = c.Content
		}
	}

	// Test individual embedding (first 5 only)
	fmt.Printf("\n4. Individual embedding times (first 5):\n")
	for i := 0; i < 5 && i < len(texts); i++ {
		t := time.Now()
		_, err := embedder.Embed(ctx, texts[i])
		if err != nil {
			fmt.Printf("   Chunk %d: ERROR - %v\n", i+1, err)
		} else {
			fmt.Printf("   Chunk %d (%d tokens): %v\n", i+1, chunk.EstimateTokens(texts[i]), time.Since(t))
		}
	}

	// Test batch embedding (all chunks at once)
	fmt.Printf("\n5. BATCH embedding (%d chunks in one request):\n", len(chunks))
	t4 := time.Now()
	_, err = embedder.EmbedBatch(ctx, texts)
	batchTime := time.Since(t4)
	if err != nil {
		fmt.Printf("   ERROR: %v\n", err)
	} else {
		fmt.Printf("   Total batch time: %v\n", batchTime)
		fmt.Printf("   Avg per chunk: %v\n", batchTime/time.Duration(len(chunks)))
	}

	// Summary
	fmt.Printf("\n=== SUMMARY ===\n")
	total := readTime + chunkTime + batchTime
	fmt.Printf("File read:    %v (%.1f%%)\n", readTime, float64(readTime)/float64(total)*100)
	fmt.Printf("Chunking:     %v (%.1f%%)\n", chunkTime, float64(chunkTime)/float64(total)*100)
	fmt.Printf("Batch embed:  %v (%.1f%%)\n", batchTime, float64(batchTime)/float64(total)*100)
	fmt.Printf("Total:        %v\n", total)
}
