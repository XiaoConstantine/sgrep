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
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/chunk"
	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/fsnotify/fsnotify"
)

// IndexConfig holds indexer configuration.
type IndexConfig struct {
	Workers          int                    // Number of parallel workers (default: 2 * NumCPU, capped at 16)
	EmbedConcurrency int                    // Concurrent embedding requests per batch (default: 8)
	Quantization     store.QuantizationMode // Vector quantization mode (none, int8, binary)
}

// DefaultIndexConfig returns sensible defaults for indexing.
func DefaultIndexConfig() *IndexConfig {
	workers := runtime.NumCPU() * 2
	if workers < 4 {
		workers = 4
	}
	if workers > 16 {
		workers = 16
	}
	return &IndexConfig{
		Workers:          workers,
		EmbedConcurrency: 8,
		Quantization:     store.QuantizeInt8, // Default to int8 for 4x storage savings
	}
}

// Indexer handles file indexing.
type Indexer struct {
	rootPath  string
	store     store.Storer
	embedder  *embed.Embedder
	chunkCfg  *chunk.Config
	indexCfg  *IndexConfig
	ignore    *IgnoreRules
	processed int64
	errors    int64
}

// New creates a new indexer for the given path with default configuration.
func New(path string) (*Indexer, error) {
	return NewWithConfig(path, nil)
}

// NewWithConfig creates a new indexer with custom configuration.
func NewWithConfig(path string, cfg *IndexConfig) (*Indexer, error) {
	if cfg == nil {
		cfg = DefaultIndexConfig()
	}

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

	// Open store with buffered writes and adaptive search
	dbPath := filepath.Join(repoDir, "index.db")
	s, err := store.OpenBuffered(dbPath, store.WithBufferedQuantization(cfg.Quantization))
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
		indexCfg: cfg,
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

	// Process files with worker pool using configured worker count
	numWorkers := idx.indexCfg.Workers

	fileChan := make(chan string, len(files))
	// Channel for documents ready to be stored (decouples embedding from DB writes)
	docChan := make(chan []*store.Document, numWorkers*2)
	var wg sync.WaitGroup
	var writerWg sync.WaitGroup

	// Start single DB writer goroutine - serializes all writes, eliminates lock contention
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		for docs := range docChan {
			if err := idx.store.StoreBatch(ctx, docs); err != nil {
				atomic.AddInt64(&idx.errors, 1)
				fmt.Fprintf(os.Stderr, "Error storing batch: %v\n", err)
			}
		}
	}()

	// Start embedding workers - they only do read/chunk/embed, then send to docChan
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range fileChan {
				docs, err := idx.prepareFile(ctx, path)
				if err != nil {
					atomic.AddInt64(&idx.errors, 1)
					fmt.Fprintf(os.Stderr, "Error indexing %s: %v\n", path, err)
				} else if len(docs) > 0 {
					docChan <- docs // Send to writer (non-blocking due to buffered channel)
					atomic.AddInt64(&idx.processed, 1)
				} else {
					atomic.AddInt64(&idx.processed, 1) // Empty file or skipped
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

	// Wait for all embedding workers to finish
	wg.Wait()
	// Close doc channel to signal writer to finish
	close(docChan)
	// Wait for writer to finish
	writerWg.Wait()

	// Flush any remaining buffered embeddings
	if err := store.FlushIfNeeded(ctx, idx.store); err != nil {
		return fmt.Errorf("failed to flush embeddings: %w", err)
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\rIndexed %d files in %v (%d errors)\n",
		idx.processed, elapsed.Round(time.Millisecond), idx.errors)

	return nil
}

// maxEmbedTokens is the safe limit for embedding input (leaving buffer under 1500 limit)
const maxEmbedTokens = 1400

// prepareFile reads, chunks, and embeds a file, returning documents ready for storage.
// This does NOT write to the database - that's handled by the single writer goroutine.
func (idx *Indexer) prepareFile(ctx context.Context, path string) ([]*store.Document, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Skip empty or very large files
	if len(content) == 0 || len(content) > 1<<20 { // 1MB limit
		return nil, nil
	}

	// Chunk the file
	relPath, _ := filepath.Rel(idx.rootPath, path)
	chunks, err := chunk.ChunkFile(relPath, string(content), idx.chunkCfg)
	if err != nil {
		return nil, err
	}

	if len(chunks) == 0 {
		return nil, nil
	}

	// Validate and re-chunk any oversized chunks
	chunks = idx.validateAndRechunk(chunks)

	// Detect if this is a test file
	isTest := isTestFile(relPath)

	// Prepare embedding texts
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		if c.Description != "" {
			texts[i] = c.Description + "\n\n" + c.Content
		} else {
			texts[i] = c.Content
		}
	}

	// Generate embeddings in batch (concurrent with retry)
	embeddings, err := idx.embedBatchWithRetry(ctx, texts, 3)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	// Build documents with embeddings
	docs := make([]*store.Document, 0, len(chunks))
	for i, c := range chunks {
		doc := &store.Document{
			ID:        fmt.Sprintf("%s:chunk_%d", relPath, i+1),
			FilePath:  relPath,
			Content:   c.Content,
			StartLine: c.StartLine,
			EndLine:   c.EndLine,
			Embedding: embeddings[i],
			IsTest:    isTest,
			Metadata: map[string]string{
				"description": c.Description,
			},
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// indexFile is a convenience wrapper for single-file indexing (used by Watch).
func (idx *Indexer) indexFile(ctx context.Context, path string) error {
	docs, err := idx.prepareFile(ctx, path)
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return nil
	}
	return idx.store.StoreBatch(ctx, docs)
}

// validateAndRechunk checks chunks for token limit compliance and re-chunks oversized ones.
func (idx *Indexer) validateAndRechunk(chunks []chunk.Chunk) []chunk.Chunk {
	var result []chunk.Chunk

	for _, c := range chunks {
		// Calculate total tokens including description
		totalText := c.Content
		if c.Description != "" {
			totalText = c.Description + "\n\n" + c.Content
		}
		tokens := chunk.EstimateTokens(totalText)

		if tokens <= maxEmbedTokens {
			result = append(result, c)
			continue
		}

		// Re-chunk this oversized chunk with a smaller limit
		// Use a conservative limit that accounts for description
		smallerCfg := &chunk.Config{
			MaxTokens:    maxEmbedTokens - chunk.EstimateTokens(c.Description) - 20,
			ContextLines: idx.chunkCfg.ContextLines,
			Overlap:      idx.chunkCfg.Overlap,
		}
		if smallerCfg.MaxTokens < 100 {
			smallerCfg.MaxTokens = 100
		}

		// Re-chunk the content
		subChunks, err := chunk.ChunkFile(c.FilePath, c.Content, smallerCfg)
		if err != nil || len(subChunks) == 0 {
			// Fallback: just use original (will be truncated by retry logic)
			result = append(result, c)
			continue
		}

		// Preserve original description with part suffix
		for i, sc := range subChunks {
			sc.Description = c.Description + fmt.Sprintf(" (part %d)", i+1)
			sc.StartLine = c.StartLine + sc.StartLine - 1
			sc.EndLine = c.StartLine + sc.EndLine - 1
			result = append(result, sc)
		}
	}

	return result
}

// embedBatchWithRetry embeds multiple texts concurrently with retry logic.
// It first tries batch embedding, then falls back to individual retry on failures.
func (idx *Indexer) embedBatchWithRetry(ctx context.Context, texts []string, maxRetries int) ([][]float32, error) {
	// First try batch embedding (concurrent with semaphore)
	embeddings, err := idx.embedder.EmbedBatch(ctx, texts)
	if err == nil {
		return embeddings, nil
	}

	// If batch failed, fall back to individual embedding with retry
	// This handles cases where only some texts are problematic
	results := make([][]float32, len(texts))
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error

	// Use semaphore for concurrency control (same as EmbedBatch)
	sem := make(chan struct{}, 8)

	for i, text := range texts {
		wg.Add(1)
		go func(index int, t string, embedder *Indexer) {
			defer wg.Done()

			sem <- struct{}{}
			defer func() { <-sem }()

			emb, err := embedder.embedWithRetry(ctx, t, maxRetries)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}

			mu.Lock()
			results[index] = emb
			mu.Unlock()
		}(i, text, idx)
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return results, nil
}

// embedWithRetry attempts to embed text, retrying with truncation on overflow errors.
func (idx *Indexer) embedWithRetry(ctx context.Context, text string, maxRetries int) ([]float32, error) {
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		embedding, err := idx.embedder.Embed(ctx, text)
		if err == nil {
			return embedding, nil
		}

		lastErr = err

		// Check if error is due to input size (llama.cpp returns 500 for oversized input)
		errStr := err.Error()
		if !strings.Contains(errStr, "too large") && !strings.Contains(errStr, "500") {
			return nil, err // Non-recoverable error
		}

		if attempt == maxRetries {
			break
		}

		// Truncate by 25% each retry
		truncateRatio := 0.75 - (float64(attempt) * 0.1)
		if truncateRatio < 0.3 {
			truncateRatio = 0.3
		}
		maxChars := int(float64(len(text)) * truncateRatio)
		text = truncateAtBoundary(text, maxChars)

		// Log the retry (to stderr so it doesn't mess up progress output)
		fmt.Fprintf(os.Stderr, "Retry %d: truncated input to %d chars\n", attempt+1, len(text))
	}

	return nil, fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr)
}

// truncateAtBoundary truncates text to maxChars, trying to break at line boundaries.
func truncateAtBoundary(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}

	truncated := text[:maxChars]

	// Try to break at line boundary (prefer 75% of way through)
	if idx := strings.LastIndex(truncated, "\n"); idx > maxChars*3/4 {
		return truncated[:idx]
	}

	// Try to break at word boundary (prefer 50% of way through)
	if idx := strings.LastIndex(truncated, " "); idx > maxChars/2 {
		return truncated[:idx]
	}

	return truncated
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

// isTestFile returns true if the file is a test file based on naming conventions.
func isTestFile(path string) bool {
	base := filepath.Base(path)
	lower := strings.ToLower(base)

	// Check common test file patterns
	testSuffixes := []string{
		"_test.go",
		".test.ts", ".test.tsx", ".test.js", ".test.jsx",
		".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
		"_test.py", "_spec.rb",
		"test.java", "tests.java",
		"_test.rs",
	}
	for _, suffix := range testSuffixes {
		if strings.HasSuffix(lower, suffix) {
			return true
		}
	}

	// Check test_*.py pattern (Python convention)
	if strings.HasPrefix(lower, "test_") && strings.HasSuffix(lower, ".py") {
		return true
	}

	// Check if file is in a test directory
	dir := filepath.Dir(path)
	testDirs := []string{"_tests", "__tests__", "tests", "test", "spec", "specs"}
	for _, td := range testDirs {
		if strings.Contains(dir, string(filepath.Separator)+td+string(filepath.Separator)) ||
			strings.HasSuffix(dir, string(filepath.Separator)+td) {
			return true
		}
	}

	return false
}
