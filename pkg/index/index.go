package index

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/chunk"
	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/fsnotify/fsnotify"
)

// Indexer handles file indexing.
type Indexer struct {
	rootPath  string
	store     store.Storer
	embedder  *embed.Embedder
	chunkCfg  *chunk.Config
	ignore    *IgnoreRules
	processed int64
	errors    int64
}

// New creates a new indexer for the given path.
func New(path string) (*Indexer, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	// Get sgrep home directory
	sgrepHome, err := getSgrepHome()
	if err != nil {
		return nil, err
	}

	// Create repo-specific subdirectory based on path hash
	repoID := hashPath(absPath)
	repoDir := filepath.Join(sgrepHome, "repos", repoID)
	if err := os.MkdirAll(repoDir, 0755); err != nil {
		return nil, err
	}

	// Store repo metadata
	if err := writeRepoMetadata(repoDir, absPath); err != nil {
		return nil, err
	}

	// Open store (use in-memory search for speed)
	dbPath := filepath.Join(repoDir, "index.db")
	s, err := store.OpenInMem(dbPath)
	if err != nil {
		return nil, err
	}

	// Load ignore rules
	ignore := NewIgnoreRules(absPath)

	return &Indexer{
		rootPath: absPath,
		store:    s,
		embedder: embed.New(),
		chunkCfg: chunk.DefaultConfig(),
		ignore:   ignore,
	}, nil
}

// getSgrepHome returns the sgrep home directory (~/.sgrep).
func getSgrepHome() (string, error) {
	// Check SGREP_HOME env var first
	if home := os.Getenv("SGREP_HOME"); home != "" {
		return home, nil
	}

	// Default to ~/.sgrep
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	return filepath.Join(homeDir, ".sgrep"), nil
}

// hashPath creates a short hash of a path for directory naming.
func hashPath(path string) string {
	// Use first 12 chars of SHA256 for uniqueness
	h := sha256.Sum256([]byte(path))
	return fmt.Sprintf("%x", h[:6])
}

// writeRepoMetadata stores metadata about the indexed repo.
func writeRepoMetadata(repoDir, repoPath string) error {
	metadata := map[string]interface{}{
		"path":       repoPath,
		"indexed_at": time.Now().Format(time.RFC3339),
	}

	data, err := json.Marshal(metadata)
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(repoDir, "metadata.json"), data, 0644)
}

// Index indexes all files in the root path.
func (idx *Indexer) Index(ctx context.Context) error {
	startTime := time.Now()
	fmt.Printf("Indexing %s...\n", idx.rootPath)

	// Collect files
	var files []string
	var skippedDirs, skippedFiles, nonCode int
	err := filepath.WalkDir(idx.rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip errors
		}

		// Skip directories
		if d.IsDir() {
			if idx.ignore.ShouldIgnore(path) {
				skippedDirs++
				return filepath.SkipDir
			}
			return nil
		}

		// Skip ignored files
		if idx.ignore.ShouldIgnore(path) {
			skippedFiles++
			return nil
		}

		// Only index code files
		if isCodeFile(path) {
			files = append(files, path)
		} else {
			nonCode++
		}

		return nil
	})
	
	fmt.Printf("Skipped: %d dirs, %d files, %d non-code\n", skippedDirs, skippedFiles, nonCode)
	if err != nil {
		return err
	}

	fmt.Printf("Found %d files to index\n", len(files))

	// Process files with worker pool
	const numWorkers = 4
	fileChan := make(chan string, len(files))
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range fileChan {
				if err := idx.indexFile(ctx, path); err != nil {
					atomic.AddInt64(&idx.errors, 1)
					fmt.Fprintf(os.Stderr, "Error indexing %s: %v\n", path, err)
				} else {
					atomic.AddInt64(&idx.processed, 1)
				}

				// Progress
				processed := atomic.LoadInt64(&idx.processed)
				if processed%10 == 0 {
					fmt.Printf("\rProcessed %d/%d files...", processed, len(files))
				}
			}
		}()
	}

	// Send files
	for _, f := range files {
		fileChan <- f
	}
	close(fileChan)

	wg.Wait()

	elapsed := time.Since(startTime)
	fmt.Printf("\rIndexed %d files in %v (%d errors)\n",
		idx.processed, elapsed.Round(time.Millisecond), idx.errors)

	return nil
}

// indexFile indexes a single file.
func (idx *Indexer) indexFile(ctx context.Context, path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	// Skip empty or very large files
	if len(content) == 0 || len(content) > 1<<20 { // 1MB limit
		return nil
	}

	// Chunk the file
	relPath, _ := filepath.Rel(idx.rootPath, path)
	chunks, err := chunk.ChunkFile(relPath, string(content), idx.chunkCfg)
	if err != nil {
		return err
	}

	if len(chunks) == 0 {
		return nil
	}

	// Generate embeddings and store
	var docs []*store.Document
	for i, c := range chunks {
		// Create embedding text with description
		embeddingText := c.Content
		if c.Description != "" {
			embeddingText = c.Description + "\n\n" + c.Content
		}

		embedding, err := idx.embedder.Embed(ctx, embeddingText)
		if err != nil {
			return fmt.Errorf("embedding failed: %w", err)
		}

		doc := &store.Document{
			ID:        fmt.Sprintf("%s:chunk_%d", relPath, i+1),
			FilePath:  relPath,
			Content:   c.Content,
			StartLine: c.StartLine,
			EndLine:   c.EndLine,
			Embedding: embedding,
			Metadata: map[string]string{
				"description": c.Description,
			},
		}
		docs = append(docs, doc)
	}

	return idx.store.StoreBatch(ctx, docs)
}

// Watch watches for file changes and re-indexes.
func (idx *Indexer) Watch(ctx context.Context) error {
	// First do a full index
	if err := idx.Index(ctx); err != nil {
		return err
	}

	fmt.Println("Watching for changes... (Ctrl+C to stop)")

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer func() { _ = watcher.Close() }()

	// Add directories recursively
	err = filepath.WalkDir(idx.rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && !idx.ignore.ShouldIgnore(path) {
			return watcher.Add(path)
		}
		return nil
	})
	if err != nil {
		return err
	}

	// Debounce timer
	var debounce *time.Timer
	pendingFiles := make(map[string]bool)
	var mu sync.Mutex

	processFiles := func() {
		mu.Lock()
		files := make([]string, 0, len(pendingFiles))
		for f := range pendingFiles {
			files = append(files, f)
		}
		pendingFiles = make(map[string]bool)
		mu.Unlock()

		for _, path := range files {
			if isCodeFile(path) && !idx.ignore.ShouldIgnore(path) {
				if _, err := os.Stat(path); err == nil {
					if err := idx.indexFile(ctx, path); err != nil {
						fmt.Fprintf(os.Stderr, "Error indexing %s: %v\n", path, err)
					} else {
						relPath, _ := filepath.Rel(idx.rootPath, path)
						fmt.Printf("Indexed: %s\n", relPath)
					}
				} else {
					// File deleted
					relPath, _ := filepath.Rel(idx.rootPath, path)
					_ = idx.store.DeleteByPath(ctx, relPath)
					fmt.Printf("Removed: %s\n", relPath)
				}
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}

			mu.Lock()
			pendingFiles[event.Name] = true
			if debounce != nil {
				debounce.Stop()
			}
			debounce = time.AfterFunc(500*time.Millisecond, processFiles)
			mu.Unlock()

		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			fmt.Fprintf(os.Stderr, "Watch error: %v\n", err)
		}
	}
}

// Close closes the indexer.
func (idx *Indexer) Close() error {
	return idx.store.Close()
}

// Store returns the underlying store for direct access.
// Useful for library users who want to use the store with a custom searcher.
func (idx *Indexer) Store() (store.Storer, error) {
	return idx.store, nil
}

// IgnoreRules handles .gitignore and .sgrepignore patterns.
type IgnoreRules struct {
	patterns []string
	rootPath string
}

func NewIgnoreRules(rootPath string) *IgnoreRules {
	ir := &IgnoreRules{rootPath: rootPath}

	// Default ignores
	ir.patterns = append(ir.patterns,
		".git",
		".sgrep",
		"node_modules",
		"vendor",
		"__pycache__",
		".idea",
		".vscode",
		"dist",
		"build",
		"*.min.js",
		"*.bundle.js",
		"go.sum",
		"package-lock.json",
		"yarn.lock",
	)

	// Load .gitignore
	ir.loadIgnoreFile(filepath.Join(rootPath, ".gitignore"))

	// Load .sgrepignore
	ir.loadIgnoreFile(filepath.Join(rootPath, ".sgrepignore"))

	return ir
}

func (ir *IgnoreRules) loadIgnoreFile(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			ir.patterns = append(ir.patterns, line)
		}
	}
}

func (ir *IgnoreRules) ShouldIgnore(path string) bool {
	relPath, err := filepath.Rel(ir.rootPath, path)
	if err != nil {
		return false
	}

	// Never ignore root
	if relPath == "." {
		return false
	}

	base := filepath.Base(path)

	for _, pattern := range ir.patterns {
		// Skip patterns that would match regular files (like binary names)
		// Only match if it looks like a directory pattern or glob
		if !strings.Contains(pattern, "*") && !strings.Contains(pattern, "/") {
			// For simple names, only match hidden dirs or known dirs
			if !strings.HasPrefix(pattern, ".") && !isKnownIgnoreDir(pattern) {
				continue
			}
		}

		// Check base name exact match
		if matched, _ := filepath.Match(pattern, base); matched {
			return true
		}
		// Check if path component matches (not substring)
		parts := strings.Split(relPath, string(filepath.Separator))
		for _, part := range parts {
			if part == pattern {
				return true
			}
			if matched, _ := filepath.Match(pattern, part); matched {
				return true
			}
		}
	}

	return false
}

// isKnownIgnoreDir returns true for directory names that should always be ignored.
func isKnownIgnoreDir(name string) bool {
	knownDirs := map[string]bool{
		"node_modules": true, "vendor": true, "__pycache__": true,
		"dist": true, "build": true, ".git": true, ".sgrep": true,
		".idea": true, ".vscode": true,
	}
	return knownDirs[name]
}

func isCodeFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	codeExts := map[string]bool{
		".go":    true,
		".ts":    true,
		".tsx":   true,
		".js":    true,
		".jsx":   true,
		".py":    true,
		".rs":    true,
		".java":  true,
		".c":     true,
		".cpp":   true,
		".h":     true,
		".hpp":   true,
		".rb":    true,
		".php":   true,
		".swift": true,
		".kt":    true,
		".scala": true,
		".md":    true,
		".yaml":  true,
		".yml":   true,
		".json":  true,
		".toml":  true,
	}
	return codeExts[ext]
}
