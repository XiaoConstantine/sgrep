package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
)

func main() {
	t0 := time.Now()

	// Simulate finding index
	homeDir, _ := os.UserHomeDir()
	cwd, _ := os.Getwd()
	h := sha256.Sum256([]byte(cwd))
	indexPath := filepath.Join(homeDir, ".sgrep", "repos", fmt.Sprintf("%x", h[:6]), "index.db")
	fmt.Printf("1. Path resolution: %v\n", time.Since(t0))

	t1 := time.Now()
	s, err := store.OpenInMem(indexPath)
	if err != nil {
		panic(err)
	}
	defer func() { _ = s.Close() }()
	fmt.Printf("2. Store open + vector load: %v\n", time.Since(t1))

	t2 := time.Now()
	e := embed.New()
	fmt.Printf("3. Embedder init: %v\n", time.Since(t2))

	t3 := time.Now()
	emb, err := e.Embed(context.Background(), "error handling")
	if err != nil {
		panic(err)
	}
	fmt.Printf("4. Embedding: %v (dims=%d)\n", time.Since(t3), len(emb))

	t4 := time.Now()
	docs, _, err := s.Search(context.Background(), emb, 10, 1.5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("5. In-memory search: %v (results=%d)\n", time.Since(t4), len(docs))

	fmt.Printf("\nTotal: %v\n", time.Since(t0))
}
