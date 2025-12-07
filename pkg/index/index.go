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
	"github.com/XiaoConstantine/sgrep/pkg/util"
	"github.com/fsnotify/fsnotify"
)

// IndexConfig holds indexer configuration.
type IndexConfig struct {
	Workers          int                    // Number of parallel file readers (default: 16)
	EmbedConcurrency int                    // Concurrent embedding requests per batch (default: 8)
	EmbedBatchSize   int                    // Number of chunks to batch for embedding (default: 64)
	Quantization     store.QuantizationMode // Vector quantization mode (none, int8, binary)
	SmartSkip        bool                   // Enable smart skipping for large repos (default: true)
}

// Large repo threshold - above this we enable smart skipping
const largeRepoThreshold = 1000

// DefaultIndexConfig returns sensible defaults for indexing.
func DefaultIndexConfig() *IndexConfig {
	// Workers = parallel file readers (CPU-bound: read + chunk)
	// These don't make HTTP requests, so we can have more
	workers := 16

	// Embed concurrency - only used in fallback path when batch API fails
	embedConcurrency := 8

	// Batch size for embedding - llama.cpp server has 16 parallel slots
	// Larger batches = better GPU/CPU utilization, fewer HTTP round trips
	embedBatchSize := 128

	return &IndexConfig{
		Workers:          workers,
		EmbedConcurrency: embedConcurrency,
		EmbedBatchSize:   embedBatchSize,
		Quantization:     store.QuantizeInt8, // Default to int8 for 4x storage savings
		SmartSkip:        true,               // Enable smart skipping for large repos
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

	// Open store with appropriate backend (sqlite-vec or libsql based on build tags)
	dbPath := filepath.Join(repoDir, "index.db")
	s, err := store.OpenDefault(dbPath, cfg.Quantization)
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

// chunkItem holds a chunk pending embedding, along with metadata to reconstruct the document.
type chunkItem struct {
	filePath   string
	chunkIndex int
	text       string // Text to embed (description + content)
	chunk      chunk.Chunk
	isTest     bool
}

// Index indexes all files in the root path.
func (idx *Indexer) Index(ctx context.Context) error {
	startTime := time.Now()
	debugLevel := util.GetDebugLevel()
	stats := util.NewTimingStats(debugLevel)

	fmt.Printf("Indexing %s...\n", idx.rootPath)
	util.Debugf(util.DebugSummary, "Indexing %s", idx.rootPath)

	// Collect files
	collectTimer := util.NewTimer("file_collection")
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
	collectDuration := collectTimer.Stop()
	stats.RecordStage("file_collection", collectDuration, int64(len(files)))
	util.Debugf(util.DebugSummary, "File collection: %d files in %v", len(files), collectDuration.Round(time.Millisecond))

	fmt.Printf("Skipped: %d dirs, %d files, %d non-code\n", skippedDirs, skippedFiles, nonCode)
	if err != nil {
		return err
	}

	// Smart skip for large repos
	if idx.indexCfg.SmartSkip && len(files) > largeRepoThreshold {
		filterTimer := util.NewTimer("smart_filter")
		originalCount := len(files)
		files = idx.smartFilter(files)
		skipped := originalCount - len(files)
		filterDuration := filterTimer.Stop()
		stats.RecordStage("smart_filter", filterDuration, int64(skipped))
		if skipped > 0 {
			fmt.Printf("Smart skip: filtered %d files (tests, generated, vendored)\n", skipped)
			util.Debugf(util.DebugSummary, "Smart filter: skipped %d files in %v", skipped, filterDuration.Round(time.Millisecond))
		}
	}

	fmt.Printf("Found %d files to index\n", len(files))

	// === Three-stage pipeline with global batching ===
	// Stage 1: File readers (parallel) - read files and chunk them
	// Stage 2: Embedding batcher (single goroutine) - collect chunks and batch embed
	// Stage 3: DB writer (single goroutine) - write documents to SQLite

	numWorkers := idx.indexCfg.Workers
	batchSize := idx.indexCfg.EmbedBatchSize
	if batchSize == 0 {
		batchSize = 64
	}

	fileChan := make(chan string, len(files))
	chunkChan := make(chan []chunkItem, numWorkers*2)   // Chunks from file readers
	docChan := make(chan []*store.Document, 256)        // Documents ready for storage (large buffer)

	var readerWg sync.WaitGroup
	var batcherWg sync.WaitGroup
	var writerWg sync.WaitGroup

	// Stage 3: Single DB writer goroutine
	// Note: This goroutine exits when docChan is closed (done after embedWg.Wait())
	writerWg.Go(func() {
		for docs := range docChan {
			writeTimer := util.NewTimer("db_write")
			if err := idx.store.StoreBatch(ctx, docs); err != nil {
				atomic.AddInt64(&idx.errors, 1)
				fmt.Fprintf(os.Stderr, "Error storing batch: %v\n", err)
			}
			writeDuration := writeTimer.Stop()
			stats.RecordOp("db_write", writeDuration, int64(len(docs)))
		}
	})

	// Stage 2: Single batcher goroutine - collects chunks and batch embeds them
	batcherWg.Go(func() {
		defer close(docChan) // Close docChan when batcher is done

		var pendingChunks []chunkItem

		for chunks := range chunkChan {
			pendingChunks = append(pendingChunks, chunks...)

			// Flush when batch is full
			for len(pendingChunks) >= batchSize {
				batch := pendingChunks[:batchSize]
				pendingChunks = pendingChunks[batchSize:]

				texts := make([]string, len(batch))
				for i, item := range batch {
					texts[i] = item.text
				}

				embedTimer := util.NewTimer("embedding")
				embeddings, err := idx.embedBatchWithRetry(ctx, texts, 3)
				embedDuration := embedTimer.Stop()
				stats.RecordOp("embedding", embedDuration, int64(len(texts)))

				if err != nil {
					atomic.AddInt64(&idx.errors, int64(len(batch)))
					fmt.Fprintf(os.Stderr, "Batch embedding failed: %v\n", err)
					continue
				}

				util.Debugf(util.DebugDetailed, "Embedded %d chunks in %v",
					len(texts), embedDuration.Round(time.Millisecond))

				docs := make([]*store.Document, len(batch))
				for i, item := range batch {
					docs[i] = &store.Document{
						ID:        fmt.Sprintf("%s:chunk_%d", item.filePath, item.chunkIndex+1),
						FilePath:  item.filePath,
						Content:   item.chunk.Content,
						StartLine: item.chunk.StartLine,
						EndLine:   item.chunk.EndLine,
						Embedding: embeddings[i],
						IsTest:    item.isTest,
						Metadata: map[string]string{
							"description": item.chunk.Description,
						},
					}
				}
				docChan <- docs
			}
		}

		// Flush remaining chunks
		if len(pendingChunks) > 0 {
			texts := make([]string, len(pendingChunks))
			for i, item := range pendingChunks {
				texts[i] = item.text
			}

			embedTimer := util.NewTimer("embedding")
			embeddings, err := idx.embedBatchWithRetry(ctx, texts, 3)
			embedDuration := embedTimer.Stop()
			stats.RecordOp("embedding", embedDuration, int64(len(texts)))

			if err != nil {
				atomic.AddInt64(&idx.errors, int64(len(pendingChunks)))
				fmt.Fprintf(os.Stderr, "Final batch embedding failed: %v\n", err)
			} else {
				docs := make([]*store.Document, len(pendingChunks))
				for i, item := range pendingChunks {
					docs[i] = &store.Document{
						ID:        fmt.Sprintf("%s:chunk_%d", item.filePath, item.chunkIndex+1),
						FilePath:  item.filePath,
						Content:   item.chunk.Content,
						StartLine: item.chunk.StartLine,
						EndLine:   item.chunk.EndLine,
						Embedding: embeddings[i],
						IsTest:    item.isTest,
						Metadata: map[string]string{
							"description": item.chunk.Description,
						},
					}
				}
				docChan <- docs
			}
		}
	})

	// Stage 1: File reader workers - read and chunk files, send to batcher
	for i := 0; i < numWorkers; i++ {
		readerWg.Go(func() {
			for path := range fileChan {
				readTimer := util.NewTimer("file_read")
				chunks, err := idx.readAndChunkFile(path)
				readDuration := readTimer.Stop()

				if err != nil {
					atomic.AddInt64(&idx.errors, 1)
					fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", path, err)
					continue
				}

				// Record timing (count = number of chunks produced)
				chunkCount := int64(len(chunks))
				if chunkCount == 0 {
					chunkCount = 1 // Count as 1 operation even if no chunks
				}
				stats.RecordOp("file_read", readDuration, chunkCount)

				if len(chunks) > 0 {
					chunkChan <- chunks
				}
				atomic.AddInt64(&idx.processed, 1)

				// Progress
				processed := atomic.LoadInt64(&idx.processed)
				if processed%10 == 0 {
					fmt.Printf("\rProcessed %d/%d files...", processed, len(files))
				}
			}
		})
	}

	// Send files to workers
	for _, f := range files {
		fileChan <- f
	}
	close(fileChan)

	// Wait for pipeline to complete in order
	readerWg.Wait()
	close(chunkChan) // Signal batcher that no more chunks coming
	batcherWg.Wait() // Batcher closes docChan when done
	writerWg.Wait()  // Writer exits when docChan is closed

	// Flush any remaining buffered embeddings
	flushTimer := util.NewTimer("flush")
	if err := store.FlushIfNeeded(ctx, idx.store); err != nil {
		return fmt.Errorf("failed to flush embeddings: %w", err)
	}
	flushDuration := flushTimer.Stop()
	if flushDuration > time.Millisecond {
		stats.RecordStage("flush", flushDuration, 1)
	}

	// Compute document-level embeddings (mean of chunk embeddings per file)
	fileEmbedTimer := util.NewTimer("file_embeddings")
	fileCount, err := idx.computeFileEmbeddings(ctx)
	fileEmbedDuration := fileEmbedTimer.Stop()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to compute file embeddings: %v\n", err)
	} else if fileCount > 0 {
		stats.RecordStage("file_embeddings", fileEmbedDuration, int64(fileCount))
		util.Debugf(util.DebugSummary, "Computed %d file embeddings in %v", fileCount, fileEmbedDuration.Round(time.Millisecond))
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\rIndexed %d files in %v (%d errors)\n",
		idx.processed, elapsed.Round(time.Millisecond), idx.errors)

	// Print debug summary
	if debugLevel >= util.DebugSummary {
		stats.PrintSummary()
	}

	return nil
}

// readAndChunkFile reads a file and returns chunk items ready for batching.
// This does NOT call the embedding server - just CPU-bound read and chunk work.
func (idx *Indexer) readAndChunkFile(path string) ([]chunkItem, error) {
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

	// Build chunk items
	items := make([]chunkItem, len(chunks))
	for i, c := range chunks {
		text := c.Content
		if c.Description != "" {
			text = c.Description + "\n\n" + c.Content
		}
		items[i] = chunkItem{
			filePath:   relPath,
			chunkIndex: i,
			text:       text,
			chunk:      c,
			isTest:     isTest,
		}
	}

	return items, nil
}

// maxEmbedTokens is the safe limit for embedding input.
// Server context per slot is 2048, use 1500 to leave large buffer for token estimation variance.
const maxEmbedTokens = 1500

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

// computeFileEmbeddings computes document-level embeddings by averaging chunk embeddings.
// This enables document-level search for queries like "what does this repo do".
func (idx *Indexer) computeFileEmbeddings(ctx context.Context) (int, error) {
	// Check if store supports computing file embeddings
	computer, ok := idx.store.(store.FileEmbeddingComputer)
	if !ok {
		return 0, nil // Store doesn't support file embeddings
	}

	return computer.ComputeAndStoreFileEmbeddings(ctx)
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

	// Use semaphore for concurrency control (matches EmbedConcurrency config)
	sem := make(chan struct{}, idx.indexCfg.EmbedConcurrency)

	for i, text := range texts {
		i, text := i, text // Capture loop variables
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()

			emb, err := idx.embedWithRetry(ctx, text, maxRetries)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}

			mu.Lock()
			results[i] = emb
			mu.Unlock()
		})
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

// smartFilter filters files for large repos to speed up indexing.
// It removes test files, generated files, and low-value content.
func (idx *Indexer) smartFilter(files []string) []string {
	result := make([]string, 0, len(files)/2)

	for _, path := range files {
		if idx.shouldSmartSkip(path) {
			continue
		}
		result = append(result, path)
	}

	return result
}

// shouldSmartSkip returns true if a file should be skipped in smart mode.
func (idx *Indexer) shouldSmartSkip(path string) bool {
	relPath, _ := filepath.Rel(idx.rootPath, path)
	base := filepath.Base(path)
	ext := strings.ToLower(filepath.Ext(path))

	// Skip test files
	if strings.HasSuffix(base, "_test.go") ||
		strings.HasSuffix(base, ".test.js") ||
		strings.HasSuffix(base, ".test.ts") ||
		strings.HasSuffix(base, ".test.tsx") ||
		strings.HasSuffix(base, ".spec.js") ||
		strings.HasSuffix(base, ".spec.ts") ||
		strings.HasSuffix(base, "_test.py") ||
		strings.HasSuffix(base, "_test.rs") {
		return true
	}

	// Skip generated files
	if strings.HasSuffix(base, ".pb.go") ||
		strings.HasSuffix(base, ".pb.gw.go") ||
		strings.HasSuffix(base, ".generated.go") ||
		strings.HasSuffix(base, ".gen.go") ||
		strings.HasSuffix(base, ".mock.go") ||
		strings.HasSuffix(base, "_mock.go") ||
		strings.HasSuffix(base, "_string.go") ||
		strings.HasSuffix(base, "_enumer.go") ||
		strings.HasSuffix(base, ".d.ts") {
		return true
	}

	// Skip vendored/third-party directories
	skipDirs := []string{
		"vendor/", "third_party/", "thirdparty/", "external/",
		"testdata/", "test_data/", "fixtures/", "mocks/",
		"c-deps/", "docs/", "examples/", "benchmarks/",
	}
	for _, dir := range skipDirs {
		if strings.Contains(relPath, dir) {
			return true
		}
	}

	// Skip non-essential file types
	skipExts := map[string]bool{
		".md": true, ".txt": true, ".rst": true,
		".json": true, ".yaml": true, ".yml": true, ".toml": true,
		".sql": true, ".csv": true,
		".svg": true, ".png": true, ".jpg": true, ".gif": true,
		".wasm": true, ".map": true,
	}
	return skipExts[ext]
}
