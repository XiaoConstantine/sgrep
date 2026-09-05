package index

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/chunk"
	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
	searchpkg "github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
	"github.com/fsnotify/fsnotify"
)

// IndexConfig holds indexer configuration.
type IndexConfig struct {
	Workers                 int                    // Number of parallel file readers (default: 16)
	EmbedConcurrency        int                    // Concurrent embedding requests per batch (default: 8)
	EmbedBatchSize          int                    // Number of chunks to batch for embedding (default: 64)
	Quantization            store.QuantizationMode // Vector quantization mode (none, int8, binary)
	SmartSkip               bool                   // Enable smart skipping for large repos (default: true)
	CompactVectorStorage    bool                   // Store first-stage vectors in TQ-MSE sidecars instead of SQL blobs
	AdaptiveColBERTSegments bool                   // Enable token-aware sqrt(M) ColBERT segment budgets
	ColBERTCodec            store.ColBERTCodec     // Late interaction segment codec (tqmse, int8, pq6)
	ExactVectorLimit        int                    // Keep exact float32 mmap vectors at or below this chunk count
}

// Large repo threshold - above this we enable smart skipping
const largeRepoThreshold = 1000

const adaptiveColBERTPoolMinSim = 0.90

const (
	colbertPQSampleSize                = 50000
	colbertPQMinTrainingVectors        = 256
	colbertPQTrainIterations           = 25
	colbertPQMinWorthwhileSegments     = 5000
	colbertTQMSEBits                   = 4
	colbertTQMSESeed                   = 42
	colbertPreindexChunkFetchBatchSize = 128
	colbertRewriteChunkBatchSize       = 128
)

func colbertEmbeddingDims() int {
	if v := os.Getenv("SGREP_DIMS"); v != "" {
		var dims int
		if _, err := fmt.Sscanf(v, "%d", &dims); err == nil && dims > 0 {
			return dims
		}
	}
	return 768
}

func newDefaultColBERTTQMSEQuantizer() (*util.TQMSEQuantizer, error) {
	return util.NewTQMSEQuantizer(util.TQMSEConfig{
		Dims: colbertEmbeddingDims(),
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	})
}

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
		Workers:                 workers,
		EmbedConcurrency:        embedConcurrency,
		EmbedBatchSize:          embedBatchSize,
		Quantization:            store.QuantizeInt8, // Default to int8 for 4x storage savings
		SmartSkip:               true,               // Enable smart skipping for large repos
		CompactVectorStorage:    true,
		AdaptiveColBERTSegments: false,
		ColBERTCodec:            store.ColBERTCodecUnspecified,
		ExactVectorLimit:        20000,
	}
}

// Indexer handles file indexing.
type Indexer struct {
	rootPath     string
	repoDir      string // Directory containing index files (e.g., ~/.sgrep/repos/<hash>)
	store        store.Storer
	embedder     *embed.Embedder
	chunkCfg     *chunk.Config
	indexCfg     *IndexConfig
	ignore       *IgnoreRules
	colbertCodec store.ColBERTCodec
	colbertPQ    *util.ProductQuantizer
	colbertTQMSE *util.TQMSEQuantizer
	compactTQ    *compactVectorCollector
	tqChunks     int
	tqFiles      int
	tqBuilt      bool
	processed    atomic.Int64
	errors       atomic.Int64
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

	// Repository metadata is marked incomplete by Index and only finalized after
	// a successful first-stage rebuild, so failed reindexes cannot masquerade as
	// compatible with a new embedding format.

	// Open store with appropriate backend (sqlite-vec or libsql based on build tags)
	dbPath := filepath.Join(repoDir, "index.db")
	s, err := store.OpenDefault(dbPath, cfg.Quantization)
	if err != nil {
		return nil, err
	}
	existingCodec := store.ColBERTCodecUnspecified
	var existingPQ *util.ProductQuantizer
	var existingTQMSE *util.TQMSEQuantizer
	if provider, ok := s.(store.ColBERTMetadataProvider); ok {
		existingCodec = provider.ColBERTCodec()
		existingPQ = provider.ProductQuantizer()
		existingTQMSE = provider.TQMSEQuantizer()
	}
	effectiveCodec := store.ResolveColBERTCodec(cfg.ColBERTCodec, existingCodec)
	if effectiveCodec != store.ColBERTCodecPQ6 {
		existingPQ = nil
	}
	if effectiveCodec != store.ColBERTCodecTQMSE {
		existingTQMSE = nil
	}

	// Load ignore rules
	ignore := NewIgnoreRules(absPath)

	return &Indexer{
		rootPath:     absPath,
		repoDir:      repoDir,
		store:        s,
		embedder:     embed.New(),
		chunkCfg:     chunk.DefaultConfig(),
		indexCfg:     cfg,
		ignore:       ignore,
		colbertCodec: effectiveCodec,
		colbertPQ:    existingPQ,
		colbertTQMSE: existingTQMSE,
	}, nil
}

// RepoDir returns the directory containing index files.
func (idx *Indexer) RepoDir() string {
	return idx.repoDir
}

// Checkpoint flushes store sidecar state when the backend supports it.
func (idx *Indexer) Checkpoint(ctx context.Context) error {
	return store.CheckpointIfNeeded(ctx, idx.store)
}

// RebuildTQVectorStore checkpoints SQL state, then refreshes compact
// first-stage dense vector artifacts from active chunk and file embeddings.
func (idx *Indexer) RebuildTQVectorStore(ctx context.Context) (int, error) {
	if err := idx.Checkpoint(ctx); err != nil {
		return 0, fmt.Errorf("checkpoint before TQ-MSE export: %w", err)
	}
	chunkCount, err := idx.ExportVectorsToTQ(ctx, idx.repoDir)
	if err != nil {
		return 0, err
	}
	if _, err := idx.ExportFileVectorsToTQ(ctx, idx.repoDir); err != nil {
		return 0, err
	}
	return chunkCount, nil
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
	return writeRepoMetadataState(repoDir, repoPath, true)
}

func writeRepoMetadataState(repoDir, repoPath string, complete bool) error {
	metadata := map[string]interface{}{
		"path":                     repoPath,
		"indexed_at":               time.Now().Format(time.RFC3339),
		"embedding_format_version": modelcfg.EmbeddingFormatVersion,
		"context_tokens":           modelcfg.ContextTokens(),
		"complete":                 complete,
	}

	data, err := json.Marshal(metadata)
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(repoDir, "metadata.json"), data, 0644)
}

// ValidateRepoMetadata rejects indexes whose document embeddings are
// incompatible with the current retrieval query format.
func ValidateRepoMetadata(repoDir string) error {
	data, err := os.ReadFile(filepath.Join(repoDir, "metadata.json"))
	if err != nil {
		return fmt.Errorf("read index metadata: %w", err)
	}
	var metadata struct {
		EmbeddingFormatVersion int  `json:"embedding_format_version"`
		ContextTokens          int  `json:"context_tokens"`
		Complete               bool `json:"complete"`
	}
	if err := json.Unmarshal(data, &metadata); err != nil {
		return fmt.Errorf("parse index metadata: %w", err)
	}
	if !metadata.Complete {
		return fmt.Errorf("index rebuild is incomplete; run 'sgrep index .'")
	}
	if metadata.EmbeddingFormatVersion != modelcfg.EmbeddingFormatVersion {
		return fmt.Errorf("index embedding format is version %d, need %d; run 'sgrep index .'", metadata.EmbeddingFormatVersion, modelcfg.EmbeddingFormatVersion)
	}
	if metadata.ContextTokens != modelcfg.ContextTokens() {
		return fmt.Errorf("index uses %d context tokens, current configuration uses %d; run 'sgrep index .'", metadata.ContextTokens, modelcfg.ContextTokens())
	}
	return nil
}

// chunkItem holds a chunk pending embedding, along with metadata to reconstruct the document.
type chunkItem struct {
	filePath   string
	chunkIndex int
	text       string // Text to embed (description + content)
	chunk      chunk.Chunk
	isTest     bool
}

type compactVectorCollector struct {
	chunks       *store.TQVectorAccumulator
	files        map[string]*compactFileVector
	exactIDs     []string
	exactVectors [][]float32
	exactLimit   int
	exactEnabled bool
}

type compactFileVector struct {
	sum        []float32
	chunkCount int
}

func newCompactVectorCollector() (*compactVectorCollector, error) {
	opts := store.TQVectorBuildOptions{
		Dims: colbertEmbeddingDims(),
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	}
	chunks, err := store.NewTQVectorAccumulator(opts)
	if err != nil {
		return nil, err
	}
	return &compactVectorCollector{
		chunks:       chunks,
		files:        make(map[string]*compactFileVector),
		exactLimit:   20000,
		exactEnabled: true,
	}, nil
}

func (c *compactVectorCollector) AddDocuments(docs []*store.Document) error {
	for _, doc := range docs {
		if err := c.chunks.Add(doc.ID, doc.Embedding); err != nil {
			return err
		}
		norm := util.NormalizeVectorCopy(doc.Embedding)
		if c.exactEnabled {
			if c.exactLimit > 0 && len(c.exactIDs) < c.exactLimit {
				c.exactIDs = append(c.exactIDs, doc.ID)
				c.exactVectors = append(c.exactVectors, norm)
			} else {
				c.exactEnabled = false
				c.exactIDs = nil
				c.exactVectors = nil
			}
		}
		file := c.files[doc.FilePath]
		if file == nil {
			file = &compactFileVector{sum: make([]float32, len(norm))}
			c.files[doc.FilePath] = file
		}
		if len(file.sum) != len(norm) {
			return fmt.Errorf("file %s has mixed vector dims: %d and %d", doc.FilePath, len(file.sum), len(norm))
		}
		for i, v := range norm {
			file.sum[i] += v
		}
		file.chunkCount++
	}
	return nil
}

func (c *compactVectorCollector) Write(ctx context.Context, repoDir string) (int, int, error) {
	chunkCount, err := c.chunks.WriteChunkStore(ctx, repoDir)
	if err != nil {
		return 0, 0, err
	}

	fileAcc, err := store.NewTQVectorAccumulator(store.TQVectorBuildOptions{
		Dims: colbertEmbeddingDims(),
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	})
	if err != nil {
		return 0, 0, err
	}
	for filePath, file := range c.files {
		if file.chunkCount == 0 {
			continue
		}
		mean := make([]float32, len(file.sum))
		scale := float32(1.0 / float64(file.chunkCount))
		for i, v := range file.sum {
			mean[i] = v * scale
		}
		if err := fileAcc.Add(filePath, mean); err != nil {
			return 0, 0, err
		}
	}
	fileCount, err := fileAcc.WriteFileStore(ctx, repoDir)
	if err != nil {
		return 0, 0, err
	}
	if c.exactEnabled && len(c.exactIDs) == chunkCount {
		mmapStore, err := store.OpenMMapVectorStore(repoDir, colbertEmbeddingDims())
		if err != nil {
			return 0, 0, fmt.Errorf("open exact vector artifact: %w", err)
		}
		mmapStore.BeginWrite()
		for i, id := range c.exactIDs {
			mmapStore.WriteVector(id, c.exactVectors[i])
		}
		if err := mmapStore.CommitWrite(); err != nil {
			_ = mmapStore.Close()
			return 0, 0, fmt.Errorf("write exact vector artifact: %w", err)
		}
		if err := mmapStore.Close(); err != nil {
			return 0, 0, err
		}
	} else if err := os.Remove(store.MMapVectorPath(repoDir)); err != nil && !os.IsNotExist(err) {
		return 0, 0, err
	}
	return chunkCount, fileCount, nil
}

// Index indexes all files in the root path.
func (idx *Indexer) Index(ctx context.Context) error {
	startTime := time.Now()
	if err := writeRepoMetadataState(idx.repoDir, idx.rootPath, false); err != nil {
		return fmt.Errorf("mark index rebuild incomplete: %w", err)
	}
	debugLevel := util.GetDebugLevel()
	stats := util.NewTimingStats(debugLevel)
	idx.processed.Store(0)
	idx.errors.Store(0)
	idx.tqChunks = 0
	idx.tqFiles = 0
	idx.tqBuilt = false
	idx.compactTQ = nil
	if !idx.indexCfg.CompactVectorStorage {
		// SQL-vector and watch modes must not leave an older exact artifact that
		// OpenForSearch would prefer over subsequently updated SQL/TQ vectors.
		for _, artifact := range []string{
			store.MMapVectorPath(idx.repoDir),
			store.TQVectorPath(idx.repoDir),
			store.TQFileVectorPath(idx.repoDir),
		} {
			if err := os.Remove(artifact); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("invalidate stale vector artifact %s: %w", artifact, err)
			}
		}
	}
	if idx.indexCfg.CompactVectorStorage {
		collector, err := newCompactVectorCollector()
		if err != nil {
			return fmt.Errorf("initialize compact TQ-MSE collector: %w", err)
		}
		collector.exactLimit = idx.indexCfg.ExactVectorLimit
		collector.exactEnabled = collector.exactLimit > 0
		idx.compactTQ = collector
	}

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
			if idx.ignore.ShouldIgnorePath(path, true) {
				skippedDirs++
				return filepath.SkipDir
			}
			return nil
		}

		// Skip ignored files
		if idx.ignore.ShouldIgnorePath(path, false) {
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
	chunkChan := make(chan []chunkItem, numWorkers*2) // Chunks from file readers
	docChan := make(chan []*store.Document, 256)      // Documents ready for storage (large buffer)

	var readerWg sync.WaitGroup
	var batcherWg sync.WaitGroup
	var writerWg sync.WaitGroup
	var chunkEmbedWall atomic.Int64
	var liveMu sync.Mutex
	liveIDs := make(map[string]struct{})
	livePaths := make(map[string]struct{})

	// Stage 3: Single DB writer goroutine
	// Note: This goroutine exits when docChan is closed (done after embedWg.Wait())
	writerWg.Go(func() {
		for docs := range docChan {
			writeTimer := util.NewTimer("db_write")
			err := idx.storeIndexBatch(ctx, docs)
			writeDuration := writeTimer.Stop()
			if err != nil {
				idx.errors.Add(1)
				fmt.Fprintf(os.Stderr, "Error storing batch: %v\n", err)
				continue
			}
			recordLiveDocuments(liveIDs, livePaths, &liveMu, docs)
			stats.RecordOp("db_write", writeDuration, int64(len(docs)))
		}
	})

	// Stage 2: Single batcher goroutine - collects chunks and batch embeds them
	// Note: Embedding server is typically the bottleneck, not HTTP concurrency.
	// Parallel HTTP just queues at the server, causing timeouts without speedup.
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
				chunkEmbedWall.Add(embedDuration.Nanoseconds())
				stats.RecordOp("embedding", embedDuration, int64(len(texts)))

				if err != nil {
					idx.errors.Add(int64(len(batch)))
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
							"lexical":     buildLexicalText(item.filePath, item.chunk.Description, item.chunk.Content),
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
			chunkEmbedWall.Add(embedDuration.Nanoseconds())
			stats.RecordOp("embedding", embedDuration, int64(len(texts)))

			if err != nil {
				idx.errors.Add(int64(len(pendingChunks)))
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
							"lexical":     buildLexicalText(item.filePath, item.chunk.Description, item.chunk.Content),
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
					idx.errors.Add(1)
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
				idx.processed.Add(1)

				// Progress
				processed := idx.processed.Load()
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
	fmt.Printf("Chunk embedding wall time: %v\n", time.Duration(chunkEmbedWall.Load()).Round(time.Millisecond))

	// Flush any remaining buffered embeddings
	flushTimer := util.NewTimer("flush")
	if err := store.FlushIfNeeded(ctx, idx.store); err != nil {
		return fmt.Errorf("failed to flush embeddings: %w", err)
	}
	flushDuration := flushTimer.Stop()
	if flushDuration > time.Millisecond {
		stats.RecordStage("flush", flushDuration, 1)
	}

	hadErrors := idx.errors.Load() != 0
	if !hadErrors {
		pruneTimer := util.NewTimer("prune_index")
		pruned, err := idx.pruneStaleIndex(ctx, liveIDs, livePaths)
		if err != nil {
			return fmt.Errorf("prune stale index rows: %w", err)
		}
		pruneDuration := pruneTimer.Stop()
		if pruned && pruneDuration > time.Millisecond {
			stats.RecordStage("prune_index", pruneDuration, 1)
		}
	} else {
		fmt.Fprintln(os.Stderr, "Warning: skipped stale index cleanup because indexing had errors")
	}

	if err := idx.refreshDerivedVectorArtifacts(ctx, stats, hadErrors); err != nil {
		return err
	}
	if hadErrors {
		return fmt.Errorf("index rebuild incomplete: %d files or chunks failed; previous compatible index remains unavailable", idx.errors.Load())
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\rIndexed %d files in %v (%d errors)\n",
		idx.processed.Load(), elapsed.Round(time.Millisecond), idx.errors.Load())

	// Print debug summary
	if debugLevel >= util.DebugSummary {
		stats.PrintSummary()
	}
	if err := writeRepoMetadata(idx.repoDir, idx.rootPath); err != nil {
		return fmt.Errorf("finalize index metadata: %w", err)
	}

	return nil
}

func recordLiveDocuments(liveIDs, livePaths map[string]struct{}, mu *sync.Mutex, docs []*store.Document) {
	mu.Lock()
	defer mu.Unlock()
	for _, doc := range docs {
		if doc == nil {
			continue
		}
		if doc.ID != "" {
			liveIDs[doc.ID] = struct{}{}
		}
		if doc.FilePath != "" {
			livePaths[doc.FilePath] = struct{}{}
		}
	}
}

func (idx *Indexer) pruneStaleIndex(ctx context.Context, liveIDs, livePaths map[string]struct{}) (bool, error) {
	pruner, ok := idx.store.(store.IndexPruner)
	if !ok {
		return false, nil
	}
	return true, pruner.PruneIndex(ctx, sortedSet(liveIDs), sortedSet(livePaths))
}

func (idx *Indexer) refreshDerivedVectorArtifacts(ctx context.Context, stats *util.TimingStats, hadErrors bool) error {
	if idx.compactTQ != nil {
		if hadErrors {
			fmt.Fprintln(os.Stderr, "Warning: skipped compact TQ-MSE export because indexing had errors")
			return nil
		}
		return idx.writeCompactVectorStores(ctx, stats)
	}

	if hadErrors {
		fmt.Fprintln(os.Stderr, "Warning: skipped file embedding refresh because indexing had errors")
		return nil
	}

	// Compute document-level embeddings (mean of chunk embeddings per file)
	fileEmbedTimer := util.NewTimer("file_embeddings")
	fileCount, err := idx.computeFileEmbeddings(ctx)
	fileEmbedDuration := fileEmbedTimer.Stop()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to compute file embeddings: %v\n", err)
		return nil
	}
	if fileCount > 0 {
		stats.RecordStage("file_embeddings", fileEmbedDuration, int64(fileCount))
		util.Debugf(util.DebugSummary, "Computed %d file embeddings in %v", fileCount, fileEmbedDuration.Round(time.Millisecond))
	}
	return nil
}

func (idx *Indexer) writeCompactVectorStores(ctx context.Context, stats *util.TimingStats) error {
	tqTimer := util.NewTimer("compact_tq_export")
	chunkCount, fileCount, err := idx.compactTQ.Write(ctx, idx.repoDir)
	tqDuration := tqTimer.Stop()
	if err != nil {
		return fmt.Errorf("write compact TQ-MSE vector stores: %w", err)
	}
	idx.tqChunks = chunkCount
	idx.tqFiles = fileCount
	idx.tqBuilt = true
	stats.RecordStage("compact_tq_export", tqDuration, int64(chunkCount+fileCount))
	if clearer, ok := idx.store.(store.VectorStorageClearer); ok {
		clearTimer := util.NewTimer("clear_sql_vectors")
		if err := clearer.ClearVectorStorage(ctx); err != nil {
			return fmt.Errorf("clear SQL vector storage: %w", err)
		}
		stats.RecordStage("clear_sql_vectors", clearTimer.Stop(), 1)
	}
	return nil
}

func sortedSet(set map[string]struct{}) []string {
	values := make([]string, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}

func (idx *Indexer) storeIndexBatch(ctx context.Context, docs []*store.Document) error {
	if idx.compactTQ == nil {
		return idx.store.StoreBatch(ctx, docs)
	}
	if metadataStore, ok := idx.store.(store.MetadataBatchStorer); ok {
		if err := metadataStore.StoreMetadataBatch(ctx, docs); err != nil {
			return err
		}
	} else if err := idx.store.StoreBatch(ctx, docs); err != nil {
		return err
	}
	return idx.compactTQ.AddDocuments(docs)
}

// CompactVectorStoreWritten reports whether Index wrote compact TQ-MSE sidecars directly.
func (idx *Indexer) CompactVectorStoreWritten() bool {
	return idx.tqBuilt
}

// CompactVectorStoreCounts returns chunk and file vector counts from direct TQ-MSE indexing.
func (idx *Indexer) CompactVectorStoreCounts() (int, int) {
	return idx.tqChunks, idx.tqFiles
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

// maxEmbedTokens is shared with the chunker and llama.cpp per-slot context.
func maxEmbedTokens() int { return modelcfg.DocumentTokenBudget() }

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
				"lexical":     buildLexicalText(relPath, c.Description, c.Content),
			},
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

func (idx *Indexer) syncFile(ctx context.Context, path string, refreshColBERT, rebuildColBERTMMap bool) error {
	relPath, err := filepath.Rel(idx.rootPath, path)
	if err != nil {
		return err
	}

	docs, err := idx.prepareFile(ctx, path)
	if err != nil {
		return err
	}

	if err := idx.store.DeleteByPath(ctx, relPath); err != nil {
		return err
	}

	if len(docs) == 0 {
		return idx.refreshFileArtifacts(ctx, nil, refreshColBERT, rebuildColBERTMMap)
	}

	if err := idx.store.StoreBatch(ctx, docs); err != nil {
		return err
	}

	return idx.refreshFileArtifacts(ctx, docs, refreshColBERT, rebuildColBERTMMap)
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

func (idx *Indexer) refreshFileArtifacts(ctx context.Context, docs []*store.Document, refreshColBERT, rebuildColBERTMMap bool) error {
	if len(docs) > 0 {
		if err := idx.refreshFileEmbedding(ctx, docs); err != nil {
			return err
		}
	}

	if !refreshColBERT {
		return nil
	}

	segmentStore, ok := idx.store.(store.ColBERTSegmentStorer)
	if !ok {
		return nil
	}

	if len(docs) > 0 {
		if err := idx.refreshColBERTSegments(ctx, segmentStore, docs); err != nil {
			return err
		}
	}

	if !rebuildColBERTMMap {
		return nil
	}

	return idx.refreshColBERTMMap(ctx, segmentStore)
}

func (idx *Indexer) shouldRefreshColBERT(ctx context.Context) (bool, error) {
	segmentStore, ok := idx.store.(store.ColBERTSegmentStorer)
	if !ok {
		return false, nil
	}

	mmapPath := filepath.Join(idx.repoDir, "colbert_segments.mmap")
	if _, err := os.Stat(mmapPath); err == nil {
		return true, nil
	} else if !os.IsNotExist(err) {
		return false, err
	}

	return segmentStore.HasColBERTSegments(ctx)
}

func (idx *Indexer) refreshFileEmbedding(ctx context.Context, docs []*store.Document) error {
	fileStore, ok := idx.store.(store.FileEmbeddingStorer)
	if !ok || len(docs) == 0 {
		return nil
	}

	dims := len(docs[0].Embedding)
	if dims == 0 {
		return nil
	}

	meanEmb := make([]float32, dims)
	maxLine := 0
	for _, doc := range docs {
		normEmb := util.NormalizeVectorCopy(doc.Embedding)
		for i, v := range normEmb {
			meanEmb[i] += v
		}
		if doc.EndLine > maxLine {
			maxLine = doc.EndLine
		}
	}

	scale := float32(1.0 / float64(len(docs)))
	for i := range meanEmb {
		meanEmb[i] *= scale
	}

	return fileStore.StoreFileEmbedding(ctx, &store.FileEmbedding{
		FilePath:   docs[0].FilePath,
		Embedding:  meanEmb,
		ChunkCount: len(docs),
		TotalLines: maxLine,
	})
}

func (idx *Indexer) refreshColBERTSegments(ctx context.Context, segmentStore store.ColBERTSegmentStorer, docs []*store.Document) error {
	if len(docs) == 0 {
		return nil
	}
	if err := idx.ensureColBERTCodecReady(ctx, segmentStore); err != nil {
		return err
	}

	chunks := make([]store.ChunkInfo, 0, len(docs))
	for _, doc := range docs {
		description := ""
		if doc.Metadata != nil {
			description = doc.Metadata["description"]
		}
		chunks = append(chunks, store.ChunkInfo{
			ID:          doc.ID,
			Content:     doc.Content,
			Description: description,
		})
	}

	chunkSegments, err := idx.buildColBERTChunkSegments(ctx, chunks)
	if err != nil {
		return err
	}
	if len(chunkSegments) == 0 {
		return nil
	}

	return segmentStore.StoreColBERTSegmentsBatch(ctx, chunkSegments)
}

func (idx *Indexer) refreshColBERTMMap(ctx context.Context, segmentStore store.ColBERTSegmentStorer) error {
	hasSegments, err := segmentStore.HasColBERTSegments(ctx)
	if err != nil {
		return err
	}

	if !hasSegments {
		mmapPath := filepath.Join(idx.repoDir, "colbert_segments.mmap")
		if err := os.Remove(mmapPath); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}

	_, err = idx.ExportColBERTToMMap(ctx, idx.repoDir)
	return err
}

// validateAndRechunk checks chunks for token limit compliance and re-chunks oversized ones.
func (idx *Indexer) validateAndRechunk(chunks []chunk.Chunk) []chunk.Chunk {
	var result []chunk.Chunk

	for _, c := range chunks {
		// Calculate total tokens including description
		totalText := util.CombineDescriptionContent(c.Content, c.Description)
		tokens := chunk.EstimateTokens(totalText)

		if tokens <= maxEmbedTokens() {
			result = append(result, c)
			continue
		}

		// This is already a semantic chunk, possibly an incomplete function
		// fragment. Re-running ChunkFile can extract only nested callbacks and
		// silently discard the statements between them. Split the existing source
		// instead, retaining its description and absolute line ranges.
		result = append(result, chunk.SplitChunk(c, &chunk.Config{
			MaxTokens: maxEmbedTokens(),
			Overlap:   idx.chunkCfg.Overlap,
		})...)
	}

	return result
}

// embedBatchWithRetry first tries one batch, then isolates individual failures.
// The legacy name is retained for compatibility; inputs are never truncated.
func (idx *Indexer) embedBatchWithRetry(ctx context.Context, texts []string, maxRetries int) ([][]float32, error) {
	// First try batch embedding (concurrent with semaphore)
	embeddings, err := idx.embedder.EmbedDocuments(ctx, texts)
	if err == nil {
		return embeddings, nil
	}

	// If the batch failed, isolate the offending input without changing content.
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

func (idx *Indexer) embedWithRetry(ctx context.Context, text string, maxRetries int) ([]float32, error) {
	_ = maxRetries
	embedding, err := idx.embedder.EmbedDocument(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("embed document without truncation: %w", err)
	}
	return embedding, nil
}

// Watch watches for file changes and re-indexes.
func (idx *Indexer) Watch(ctx context.Context) error {
	// First do a full index
	if err := idx.Index(ctx); err != nil {
		return err
	}
	fmt.Println("Refreshing compact TQ-MSE vector stores...")
	vecCount, err := idx.RebuildTQVectorStore(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to refresh compact TQ-MSE vector stores: %v\n", err)
	} else {
		fmt.Printf("Refreshed compact TQ-MSE vector stores (%d chunk vectors)\n", vecCount)
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
		if d.IsDir() && !idx.ignore.ShouldIgnorePath(path, true) {
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
	var processMu sync.Mutex
	fatalWatchErr := make(chan error, 1)
	reportFatal := func(err error) {
		select {
		case fatalWatchErr <- err:
		default:
		}
	}

	processFiles := func() {
		processMu.Lock()
		defer processMu.Unlock()

		mu.Lock()
		files := make([]string, 0, len(pendingFiles))
		for f := range pendingFiles {
			files = append(files, f)
		}
		pendingFiles = make(map[string]bool)
		mu.Unlock()
		if len(files) == 0 {
			return
		}

		refreshColBERT, err := idx.shouldRefreshColBERT(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error checking ColBERT state for batch: %v\n", err)
			refreshColBERT = false
		}

		changed := false
		for _, path := range files {
			info, statErr := os.Stat(path)
			if isCodeFile(path) && !idx.ignore.ShouldIgnorePath(path, statErr == nil && info.IsDir()) {
				if statErr == nil {
					if err := idx.syncFile(ctx, path, refreshColBERT, false); err != nil {
						fmt.Fprintf(os.Stderr, "Error indexing %s: %v\n", path, err)
					} else {
						relPath, _ := filepath.Rel(idx.rootPath, path)
						fmt.Printf("Indexed: %s\n", relPath)
						changed = true
					}
				} else {
					// File deleted
					relPath, _ := filepath.Rel(idx.rootPath, path)
					if err := idx.store.DeleteByPath(ctx, relPath); err != nil {
						fmt.Fprintf(os.Stderr, "Error removing %s: %v\n", path, err)
						continue
					}
					fmt.Printf("Removed: %s\n", relPath)
					changed = true
				}
			}
		}

		if refreshColBERT {
			segmentStore, ok := idx.store.(store.ColBERTSegmentStorer)
			if ok {
				if err := idx.refreshColBERTMMap(ctx, segmentStore); err != nil {
					fmt.Fprintf(os.Stderr, "Error refreshing ColBERT mmap: %v\n", err)
					if removeErr := os.Remove(filepath.Join(idx.repoDir, "colbert_segments.mmap")); removeErr != nil && !os.IsNotExist(removeErr) {
						reportFatal(fmt.Errorf("invalidate stale ColBERT mmap: %w", removeErr))
						return
					}
				}
			}
		}

		if changed {
			vecCount, err := idx.RebuildTQVectorStore(ctx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error refreshing compact TQ-MSE vector stores: %v\n", err)
				for _, artifact := range []string{store.TQVectorPath(idx.repoDir), store.TQFileVectorPath(idx.repoDir)} {
					if removeErr := os.Remove(artifact); removeErr != nil && !os.IsNotExist(removeErr) {
						reportFatal(fmt.Errorf("invalidate stale vector artifact %s: %w", artifact, removeErr))
						return
					}
				}
			} else {
				fmt.Printf("Refreshed compact TQ-MSE vector stores (%d chunk vectors)\n", vecCount)
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case err := <-fatalWatchErr:
			return err
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
	rootPath   string
	mu         sync.Mutex
	loadedDirs map[string]bool
	rules      []ignoreRule
}

type ignoreRule struct {
	baseRel  string
	pattern  string
	negated  bool
	dirOnly  bool
	anchored bool
	hasSlash bool
}

func NewIgnoreRules(rootPath string) *IgnoreRules {
	rootPath = filepath.Clean(rootPath)
	ir := &IgnoreRules{
		rootPath:   rootPath,
		loadedDirs: make(map[string]bool),
	}

	// Default ignores
	ir.addRule("", ".git/")
	ir.addRule("", ".sgrep/")
	ir.addRule("", "node_modules/")
	ir.addRule("", "vendor/")
	ir.addRule("", "__pycache__/")
	ir.addRule("", ".idea/")
	ir.addRule("", ".vscode/")
	ir.addRule("", "dist/")
	ir.addRule("", "build/")
	for _, pattern := range []string{
		"*.min.js",
		"*.bundle.js",
		"go.sum",
		"package-lock.json",
		"yarn.lock",
	} {
		ir.addRule("", pattern)
	}

	ir.ensureRulesLoaded(rootPath)

	return ir
}

func (ir *IgnoreRules) addRule(baseRel, raw string) {
	if rule, ok := parseIgnoreRule(baseRel, raw); ok {
		ir.rules = append(ir.rules, rule)
	}
}

func parseIgnoreRule(baseRel, line string) (ignoreRule, bool) {
	line = strings.TrimSpace(strings.TrimRight(line, "\r"))
	if line == "" {
		return ignoreRule{}, false
	}
	if strings.HasPrefix(line, `\#`) || strings.HasPrefix(line, `\!`) {
		line = line[1:]
	}
	if strings.HasPrefix(line, "#") {
		return ignoreRule{}, false
	}

	negated := strings.HasPrefix(line, "!")
	if negated {
		line = strings.TrimPrefix(line, "!")
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return ignoreRule{}, false
	}

	dirOnly := strings.HasSuffix(line, "/")
	line = strings.TrimSuffix(line, "/")
	anchored := strings.HasPrefix(line, "/")
	line = strings.TrimPrefix(line, "/")
	line = filepath.ToSlash(filepath.Clean(line))
	if line == "." || line == "" {
		return ignoreRule{}, false
	}

	return ignoreRule{
		baseRel:  normalizeIgnoreRel(baseRel),
		pattern:  line,
		negated:  negated,
		dirOnly:  dirOnly,
		anchored: anchored,
		hasSlash: strings.Contains(line, "/"),
	}, true
}

func normalizeIgnoreRel(rel string) string {
	rel = filepath.ToSlash(filepath.Clean(rel))
	if rel == "." {
		return ""
	}
	return rel
}

func (ir *IgnoreRules) ensureRulesLoaded(targetPath string) {
	targetPath = filepath.Clean(targetPath)
	dirPath := targetPath
	if info, err := os.Stat(targetPath); err == nil && !info.IsDir() {
		dirPath = filepath.Dir(targetPath)
	}

	relDir, err := filepath.Rel(ir.rootPath, dirPath)
	if err != nil || strings.HasPrefix(relDir, "..") {
		return
	}
	relDir = normalizeIgnoreRel(relDir)

	ir.mu.Lock()
	defer ir.mu.Unlock()
	if ir.loadedDirs == nil {
		ir.loadedDirs = make(map[string]bool)
	}

	current := ir.rootPath
	ir.loadIgnoreFilesInDirLocked(current)
	if relDir == "" {
		return
	}

	for _, part := range strings.Split(relDir, "/") {
		current = filepath.Join(current, part)
		ir.loadIgnoreFilesInDirLocked(current)
	}
}

func (ir *IgnoreRules) loadIgnoreFilesInDirLocked(dir string) {
	dir = filepath.Clean(dir)
	if ir.loadedDirs[dir] {
		return
	}
	ir.loadIgnoreFileLocked(filepath.Join(dir, ".gitignore"), dir)
	ir.loadIgnoreFileLocked(filepath.Join(dir, ".sgrepignore"), dir)
	ir.loadedDirs[dir] = true
}

func (ir *IgnoreRules) loadIgnoreFileLocked(path string, baseDir string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()

	baseRel, err := filepath.Rel(ir.rootPath, baseDir)
	if err != nil {
		return
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		ir.addRule(baseRel, scanner.Text())
	}
}

func (ir *IgnoreRules) ShouldIgnore(path string) bool {
	info, err := os.Stat(path)
	return ir.ShouldIgnorePath(path, err == nil && info.IsDir())
}

func (ir *IgnoreRules) ShouldIgnorePath(path string, isDir bool) bool {
	ir.ensureRulesLoaded(path)

	relPath, err := filepath.Rel(ir.rootPath, path)
	if err != nil || strings.HasPrefix(relPath, "..") {
		return false
	}
	relPath = normalizeIgnoreRel(relPath)

	// Never ignore root
	if relPath == "." {
		return false
	}

	ir.mu.Lock()
	defer ir.mu.Unlock()

	ignored := false
	for _, rule := range ir.rules {
		if !pathWithinIgnoreBase(relPath, rule.baseRel) {
			continue
		}
		candidate := relPath
		if rule.baseRel != "" {
			if relPath == rule.baseRel {
				candidate = ""
			} else {
				candidate = strings.TrimPrefix(relPath, rule.baseRel+"/")
			}
		}
		if rule.matches(candidate, isDir) {
			ignored = !rule.negated
		}
	}

	return ignored
}

func pathWithinIgnoreBase(relPath, baseRel string) bool {
	if baseRel == "" {
		return true
	}
	return relPath == baseRel || strings.HasPrefix(relPath, baseRel+"/")
}

func (r ignoreRule) matches(candidate string, isDir bool) bool {
	if candidate == "" {
		return false
	}
	if r.dirOnly {
		for _, prefix := range directoryPrefixes(candidate, isDir) {
			if r.matchesPath(prefix) {
				return true
			}
		}
		return false
	}
	return r.matchesPath(candidate)
}

func (r ignoreRule) matchesPath(candidate string) bool {
	if candidate == "" {
		return false
	}
	candidate = filepath.ToSlash(candidate)

	if r.hasSlash {
		return matchIgnoreGlob(r.pattern, candidate)
	}
	if r.anchored {
		if strings.Contains(candidate, "/") {
			return false
		}
		return matchIgnoreGlob(r.pattern, candidate)
	}

	for _, part := range strings.Split(candidate, "/") {
		if matchIgnoreGlob(r.pattern, part) {
			return true
		}
	}
	return false
}

func directoryPrefixes(candidate string, isDir bool) []string {
	parts := strings.Split(filepath.ToSlash(candidate), "/")
	limit := len(parts)
	if !isDir && limit == 0 {
		return nil
	}

	prefixes := make([]string, 0, limit)
	for i := 1; i <= limit; i++ {
		prefixes = append(prefixes, strings.Join(parts[:i], "/"))
	}
	return prefixes
}

func matchIgnoreGlob(pattern, candidate string) bool {
	if strings.Contains(pattern, "**") {
		return doublestarMatch(pattern, candidate)
	}
	matched, err := pathpkg.Match(pattern, candidate)
	return err == nil && matched
}

func doublestarMatch(pattern, candidate string) bool {
	var re strings.Builder
	re.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		switch pattern[i] {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				re.WriteString(".*")
				i++
			} else {
				re.WriteString(`[^/]*`)
			}
		case '?':
			re.WriteString(`[^/]`)
		default:
			re.WriteString(regexp.QuoteMeta(string(pattern[i])))
		}
	}
	re.WriteString("$")
	matched, err := regexp.MatchString(re.String(), candidate)
	return err == nil && matched
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

// ComputeColBERTSegments pre-computes and stores ColBERT segment embeddings for all chunks.
// This enables fast MaxSim scoring at query time (~1-5ms vs ~100ms per doc).
// Returns the number of chunks processed and any error.
func (idx *Indexer) ComputeColBERTSegments(ctx context.Context) (int, error) {
	// Check if store supports ColBERT segments
	segmentStore, ok := idx.store.(store.ColBERTSegmentStorer)
	if !ok {
		return 0, fmt.Errorf("store does not support ColBERT segments")
	}

	if err := idx.applyPQCodecSizeGate(ctx, segmentStore); err != nil {
		return 0, err
	}

	if idx.colbertCodec == store.ColBERTCodecPQ6 && idx.colbertPQ == nil {
		hasSegments, err := segmentStore.HasColBERTSegments(ctx)
		if err != nil {
			return 0, fmt.Errorf("check existing ColBERT segments: %w", err)
		}
		if !hasSegments {
			return idx.computeFreshPQSegmentsFromScratch(ctx, segmentStore)
		}

		if err := idx.ensureColBERTCodecReady(ctx, segmentStore); err != nil {
			return 0, err
		}
		rewritten, err := idx.rewriteStoredColBERTSegmentsToPQ(ctx, segmentStore)
		if err != nil {
			return rewritten, err
		}
		if err := idx.persistColBERTMetadata(ctx); err != nil {
			return rewritten, err
		}
		return rewritten, nil
	}

	if err := idx.ensureColBERTCodecReady(ctx, segmentStore); err != nil {
		return 0, err
	}
	processed, err := idx.computeColBERTSegmentsPass(ctx, segmentStore, "Computing ColBERT segments")
	if err != nil {
		return processed, err
	}
	if idx.colbertCodec == store.ColBERTCodecPQ6 {
		if err := idx.persistColBERTMetadata(ctx); err != nil {
			return processed, err
		}
	}
	return processed, nil
}

func (idx *Indexer) applyPQCodecSizeGate(ctx context.Context, segmentStore store.ColBERTSegmentStorer) error {
	if idx.colbertCodec != store.ColBERTCodecPQ6 {
		return nil
	}

	estimatedSegments, err := idx.estimateColBERTStoredSegmentCount(ctx, segmentStore)
	if err != nil {
		return fmt.Errorf("estimate ColBERT segment count for PQ6 gate: %w", err)
	}
	if estimatedSegments >= colbertPQMinWorthwhileSegments {
		return nil
	}

	fmt.Printf("Skipping PQ6: estimated %d ColBERT segments is below the worthwhile savings threshold of %d, keeping int8\n",
		estimatedSegments, colbertPQMinWorthwhileSegments)
	idx.colbertCodec = store.ColBERTCodecInt8
	idx.colbertPQ = nil
	idx.colbertTQMSE = nil
	return nil
}

func (idx *Indexer) estimateColBERTStoredSegmentCount(ctx context.Context, segmentStore store.ColBERTSegmentStorer) (int, error) {
	const fetchBatchSize = 256

	totalSegments := 0
	offset := 0
	adaptive := idx.useAdaptiveColBERTSegments()

	for {
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			return 0, fmt.Errorf("get chunks at offset %d: %w", offset, err)
		}
		if len(chunks) == 0 {
			break
		}

		for _, chunk := range chunks {
			combined := util.CombineDescriptionContent(chunk.Content, chunk.Description)
			totalSegments += estimateStoredColBERTSegments(combined, adaptive)
			if totalSegments >= colbertPQMinWorthwhileSegments {
				return totalSegments, nil
			}
		}
		offset += len(chunks)
	}

	return totalSegments, nil
}

func estimateStoredColBERTSegments(content string, adaptive bool) int {
	if adaptive {
		return searchpkg.AdaptiveSegmentBudgetFromRawCount(len(searchpkg.DecomposeDocumentRaw(content)))
	}
	return len(searchpkg.DecomposeDocument(content))
}

func (idx *Indexer) computeColBERTSegmentsPass(ctx context.Context, segmentStore store.ColBERTSegmentStorer, progressLabel string) (int, error) {
	// Get total chunk count for progress reporting
	stats, err := idx.store.Stats(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to get stats: %w", err)
	}

	totalChunks := int(stats.Chunks)
	fmt.Printf("%s for %d chunks...\n", progressLabel, totalChunks)

	if totalChunks == 0 {
		return 0, nil
	}

	// Process chunks in paginated batches to handle large repos
	// Fetch 32 chunks at a time from DB, process them, then fetch next batch
	const fetchBatchSize = colbertPreindexChunkFetchBatchSize
	processed := 0
	offset := 0

	for {
		// Fetch next batch of chunks from database
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			return processed, fmt.Errorf("failed to get chunks at offset %d: %w", offset, err)
		}

		// No more chunks to process
		if len(chunks) == 0 {
			break
		}

		chunkSegments, err := idx.buildColBERTChunkSegments(ctx, chunks)
		if err != nil {
			return processed, fmt.Errorf("build ColBERT segments at offset %d: %w", offset, err)
		}

		if len(chunkSegments) != len(chunks) {
			return processed, fmt.Errorf("ColBERT preindex coverage mismatch at offset %d: built %d of %d chunks", offset, len(chunkSegments), len(chunks))
		}

		// Store batch in SQLite
		if err := segmentStore.StoreColBERTSegmentsBatch(ctx, chunkSegments); err != nil {
			return processed, fmt.Errorf("store ColBERT segments at offset %d: %w", offset, err)
		}

		offset += len(chunks)
		processed += len(chunks)
		fmt.Printf("Processed %d/%d chunks...\n", processed, totalChunks)
	}

	if processed != totalChunks {
		return processed, fmt.Errorf("ColBERT preindex coverage mismatch: processed %d of %d chunks", processed, totalChunks)
	}
	return processed, nil
}

func (idx *Indexer) computeFreshPQSegmentsFromScratch(ctx context.Context, segmentStore store.ColBERTSegmentStorer) (int, error) {
	stats, err := idx.store.Stats(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to get stats: %w", err)
	}

	totalChunks := int(stats.Chunks)
	fmt.Printf("Computing ColBERT float32 scratch segments for %d chunks...\n", totalChunks)
	if totalChunks == 0 {
		return 0, nil
	}

	scratch, err := newColBERTPQScratchWriter(idx.repoDir)
	if err != nil {
		return 0, fmt.Errorf("create ColBERT PQ scratch file: %w", err)
	}
	scratchPath := scratch.Path()
	defer func() { _ = os.Remove(scratchPath) }()

	const fetchBatchSize = colbertPreindexChunkFetchBatchSize
	processed := 0
	offset := 0
	rng := rand.New(rand.NewSource(42))
	sample := make([][]float32, 0, colbertPQSampleSize)
	totalVectors := 0
	scratchBuildStart := time.Now()

	for {
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			_ = scratch.Close()
			return processed, fmt.Errorf("failed to get chunks at offset %d: %w", offset, err)
		}
		if len(chunks) == 0 {
			break
		}

		chunkSegments, err := idx.buildFloat32ColBERTChunkSegments(ctx, chunks)
		if err != nil {
			_ = scratch.Close()
			return processed, fmt.Errorf("failed to build float32 ColBERT scratch segments: %w", err)
		}

		for _, chunk := range chunks {
			segments := chunkSegments[chunk.ID]
			if len(segments) == 0 {
				continue
			}
			if err := scratch.WriteChunk(chunk.ID, segments); err != nil {
				_ = scratch.Close()
				return processed, fmt.Errorf("write ColBERT PQ scratch chunk %s: %w", chunk.ID, err)
			}
			for _, seg := range segments {
				if len(seg.Embedding) == 0 {
					continue
				}
				totalVectors++
				sample = reservoirSampleEmbeddings(sample, seg.Embedding, totalVectors, colbertPQSampleSize, rng)
			}
		}

		offset += len(chunks)
		processed += len(chunks)
		fmt.Printf("Processed %d/%d chunks...\n", processed, totalChunks)
	}

	if err := scratch.Close(); err != nil {
		return processed, fmt.Errorf("close ColBERT PQ scratch file: %w", err)
	}
	fmt.Printf("ColBERT scratch build wall time: %v\n", time.Since(scratchBuildStart).Round(time.Millisecond))

	if totalVectors < colbertPQMinTrainingVectors {
		fmt.Printf("Skipping PQ6: only %d segment vectors available, keeping int8\n", totalVectors)
		idx.colbertCodec = store.ColBERTCodecInt8
		idx.colbertPQ = nil
		idx.colbertTQMSE = nil
		written, err := idx.rewriteScratchColBERTSegments(ctx, segmentStore, scratchPath, totalChunks)
		if err != nil {
			return written, err
		}
		if err := idx.persistColBERTMetadata(ctx); err != nil {
			return written, err
		}
		return written, nil
	}

	fmt.Printf("Training ColBERT PQ codebook on %d sampled vectors (%d total)...\n", len(sample), totalVectors)
	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       768,
		Subspaces:  6,
		Centroids:  256,
		Iterations: colbertPQTrainIterations,
	})
	if err != nil {
		return processed, fmt.Errorf("create colbert pq: %w", err)
	}
	trainStart := time.Now()
	if err := pq.Train(sample, colbertPQTrainIterations); err != nil {
		return processed, fmt.Errorf("train colbert pq: %w", err)
	}
	fmt.Printf("ColBERT PQ train wall time: %v\n", time.Since(trainStart).Round(time.Millisecond))

	idx.colbertPQ = pq
	idx.colbertTQMSE = nil
	written, err := idx.rewriteScratchColBERTSegments(ctx, segmentStore, scratchPath, totalChunks)
	if err != nil {
		return written, err
	}
	if err := idx.persistColBERTMetadata(ctx); err != nil {
		return written, err
	}
	return written, nil
}

func reservoirSampleEmbeddings(sample [][]float32, emb []float32, totalVectors int, sampleSize int, rng *rand.Rand) [][]float32 {
	if len(sample) < sampleSize {
		return append(sample, append([]float32(nil), emb...))
	}
	replaceIdx := rng.Intn(totalVectors)
	if replaceIdx < len(sample) {
		sample[replaceIdx] = append([]float32(nil), emb...)
	}
	return sample
}

func (idx *Indexer) rewriteScratchColBERTSegments(ctx context.Context, segmentStore store.ColBERTSegmentStorer, scratchPath string, totalChunks int) (int, error) {
	reader, err := openColBERTPQScratchReader(scratchPath)
	if err != nil {
		return 0, fmt.Errorf("open ColBERT PQ scratch reader: %w", err)
	}
	defer func() { _ = reader.Close() }()

	label := "Writing ColBERT int8 segments"
	progressVerb := "Wrote"
	if idx.colbertCodec == store.ColBERTCodecPQ6 && idx.colbertPQ != nil {
		label = "Re-encoding ColBERT segments with PQ6"
		progressVerb = "Re-encoded"
	} else if idx.colbertCodec == store.ColBERTCodecTQMSE && idx.colbertTQMSE != nil {
		label = "Encoding ColBERT segments with TQ-MSE"
		progressVerb = "Encoded"
	}
	fmt.Printf("%s for %d chunks...\n", label, totalChunks)

	const batchSize = colbertRewriteChunkBatchSize
	batch := make(map[string][]store.ColBERTSegment, batchSize)
	processed := 0
	flushBatch := func() error {
		if len(batch) == 0 {
			return nil
		}
		if err := segmentStore.StoreColBERTSegmentsBatch(ctx, batch); err != nil {
			return err
		}
		for chunkID := range batch {
			delete(batch, chunkID)
		}
		return nil
	}

	for {
		record, err := reader.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return processed, fmt.Errorf("read ColBERT PQ scratch record: %w", err)
		}
		converted, err := convertColBERTSegmentsForCodecStrict(record.ChunkID, record.Segments, idx.colbertCodec, idx.colbertPQ, idx.colbertTQMSE)
		if err != nil {
			return processed, fmt.Errorf("convert scratch chunk %s to %s: %w", record.ChunkID, idx.colbertCodec, err)
		}
		batch[record.ChunkID] = converted
		processed++
		if len(batch) >= batchSize {
			if err := flushBatch(); err != nil {
				return processed, fmt.Errorf("store ColBERT scratch batch: %w", err)
			}
			fmt.Printf("%s %d/%d chunks...\n", progressVerb, processed, totalChunks)
		}
	}

	if err := flushBatch(); err != nil {
		return processed, fmt.Errorf("store final ColBERT scratch batch: %w", err)
	}
	if processed%batchSize != 0 {
		fmt.Printf("%s %d/%d chunks...\n", progressVerb, processed, totalChunks)
	}
	return processed, nil
}

func (idx *Indexer) rewriteStoredColBERTSegmentsToPQ(ctx context.Context, segmentStore store.ColBERTSegmentStorer) (int, error) {
	if idx.colbertCodec != store.ColBERTCodecPQ6 || idx.colbertPQ == nil {
		return 0, nil
	}

	stats, err := idx.store.Stats(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to get stats for PQ rewrite: %w", err)
	}

	totalChunks := int(stats.Chunks)
	fmt.Printf("Re-encoding ColBERT segments with PQ6 for %d chunks...\n", totalChunks)
	if totalChunks == 0 {
		return 0, nil
	}

	const fetchBatchSize = colbertRewriteChunkBatchSize
	processed := 0
	offset := 0

	for {
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			return processed, fmt.Errorf("failed to fetch chunks for PQ rewrite at offset %d: %w", offset, err)
		}
		if len(chunks) == 0 {
			break
		}

		chunkIDs := make([]string, len(chunks))
		for i, chunk := range chunks {
			chunkIDs[i] = chunk.ID
		}
		segmentMap, err := segmentStore.GetColBERTSegmentsBatch(ctx, chunkIDs)
		if err != nil {
			return processed, fmt.Errorf("failed to load segments for PQ rewrite at offset %d: %w", offset, err)
		}

		encodedBatch := make(map[string][]store.ColBERTSegment, len(chunkIDs))
		for _, chunk := range chunks {
			segments := segmentMap[chunk.ID]
			if len(segments) == 0 {
				continue
			}
			encoded, err := encodeSegmentsToPQ(chunk.ID, segments, idx.colbertPQ)
			if err != nil {
				return processed, fmt.Errorf("failed to encode chunk %s for PQ rewrite: %w", chunk.ID, err)
			}
			encodedBatch[chunk.ID] = encoded
		}

		if len(encodedBatch) > 0 {
			if err := segmentStore.StoreColBERTSegmentsBatch(ctx, encodedBatch); err != nil {
				return processed, fmt.Errorf("failed to store PQ rewrite batch at offset %d: %w", offset, err)
			}
		}

		offset += len(chunks)
		processed += len(chunks)
		fmt.Printf("Re-encoded %d/%d chunks...\n", processed, totalChunks)
	}

	return processed, nil
}

// ExportColBERTToMMap exports all ColBERT segments from SQLite to an MMap file.
// This provides faster read access at query time (zero-copy memory mapping).
// Returns the number of segments exported and any error.
func (idx *Indexer) ExportColBERTToMMap(ctx context.Context, outputDir string) (int, error) {
	segmentStore, ok := idx.store.(store.ColBERTSegmentStorer)
	if !ok {
		return 0, fmt.Errorf("store does not support ColBERT segments")
	}

	// Check if segments exist
	hasSegments, err := segmentStore.HasColBERTSegments(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to check segments: %w", err)
	}
	if !hasSegments {
		return 0, fmt.Errorf("no ColBERT segments found; run with --colbert-preindex first")
	}

	tempDir, err := os.MkdirTemp(outputDir, "colbert-mmap-*")
	if err != nil {
		return 0, fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(tempDir) }()

	// Build the mmap in a temp location, then atomically replace the live file.
	mmapStore, err := store.OpenMMapSegmentStore(tempDir, colbertEmbeddingDims())
	if err != nil {
		return 0, fmt.Errorf("failed to create MMap store: %w", err)
	}
	defer func() {
		if mmapStore != nil {
			_ = mmapStore.Close()
		}
	}()
	exportCodec := store.ColBERTCodecInt8
	var exportPQ *util.ProductQuantizer
	var exportTQ *util.TQMSEQuantizer
	if provider, ok := segmentStore.(store.ColBERTMetadataProvider); ok {
		switch provider.ColBERTCodec() {
		case store.ColBERTCodecPQ6:
			exportCodec = store.ColBERTCodecPQ6
			exportPQ = provider.ProductQuantizer()
		case store.ColBERTCodecTQMSE:
			exportCodec = store.ColBERTCodecTQMSE
			exportTQ = provider.TQMSEQuantizer()
		default:
			exportCodec = store.ColBERTCodecInt8
		}
		mmapStore.SetColBERTCodec(exportCodec, exportPQ, exportTQ)
	}

	mmapStore.BeginWrite()
	totalSegments := 0
	exportedChunks := 0
	var exportTQEncoder *util.TQMSEEncoder
	if exportCodec == store.ColBERTCodecTQMSE && exportTQ != nil {
		exportTQEncoder = exportTQ.NewEncoder()
	}
	exportChunk := func(chunkID string, chunkSegments []store.ColBERTSegment) error {
		// Write segments to MMap in the artifact codec. SQLite stores may be
		// mixed after an upgrade or partial refresh.
		encodedSegments, err := convertColBERTSegmentsForCodecWithFallback(chunkID, chunkSegments, exportCodec, exportPQ, exportTQ, false, exportTQEncoder)
		if err != nil {
			return fmt.Errorf("failed to encode chunk %s for MMap export: %w", chunkID, err)
		}
		if len(encodedSegments) == 0 {
			return fmt.Errorf("chunk %s has no ColBERT segments during MMap export", chunkID)
		}
		mmapStore.WriteSegments(chunkID, encodedSegments)
		exportedChunks++
		totalSegments += len(encodedSegments)
		return nil
	}

	if exporter, ok := segmentStore.(store.ColBERTSegmentExporter); ok {
		if err := exporter.ExportColBERTSegments(ctx, exportChunk); err != nil {
			return 0, fmt.Errorf("failed to stream ColBERT segments: %w", err)
		}
	} else {
		// Generic stores retain the paginated export path.
		const fetchBatchSize = 1000
		const segmentBatchSize = 100
		offset := 0
		for {
			chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
			if err != nil {
				return 0, fmt.Errorf("failed to fetch chunks at offset %d: %w", offset, err)
			}
			if len(chunks) == 0 {
				break
			}

			for i := 0; i < len(chunks); i += segmentBatchSize {
				end := min(i+segmentBatchSize, len(chunks))
				batchChunks := chunks[i:end]
				chunkIDs := make([]string, len(batchChunks))
				for j, chunk := range batchChunks {
					chunkIDs[j] = chunk.ID
				}
				segments, err := segmentStore.GetColBERTSegmentsBatch(ctx, chunkIDs)
				if err != nil {
					return 0, fmt.Errorf("failed to load segments: %w", err)
				}
				for chunkID, chunkSegments := range segments {
					if err := exportChunk(chunkID, chunkSegments); err != nil {
						return 0, err
					}
				}
			}
			offset += len(chunks)
		}
	}

	stats, err := idx.store.Stats(ctx)
	if err != nil {
		return 0, fmt.Errorf("load expected ColBERT export coverage: %w", err)
	}
	if exportedChunks != int(stats.Chunks) {
		return 0, fmt.Errorf("ColBERT MMap coverage mismatch: exported %d of %d chunks", exportedChunks, stats.Chunks)
	}
	if err := mmapStore.CommitWrite(); err != nil {
		return 0, fmt.Errorf("failed to commit MMap: %w", err)
	}

	if err := mmapStore.Close(); err != nil {
		return 0, fmt.Errorf("failed to close temp MMap: %w", err)
	}
	mmapStore = nil

	tempPath := filepath.Join(tempDir, "colbert_segments.mmap")
	finalPath := filepath.Join(outputDir, "colbert_segments.mmap")
	if err := os.Rename(tempPath, finalPath); err != nil {
		return 0, fmt.Errorf("failed to replace MMap: %w", err)
	}

	return totalSegments, nil
}

// ExportVectorsToMMap exports all chunk vector embeddings from SQLite to an MMap file.
// This provides faster read access at query time (zero-copy memory mapping).
// Returns the number of vectors exported and any error.
func (idx *Indexer) ExportVectorsToMMap(ctx context.Context, outputDir string) (int, error) {
	// Check if store implements VectorExporter interface
	exporter, ok := idx.store.(store.VectorExporter)
	if !ok {
		return 0, fmt.Errorf("store does not support vector export")
	}

	// Get all vectors directly from store
	chunkIDs, embeddings, err := exporter.ExportAllVectors(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to export vectors: %w", err)
	}

	if len(chunkIDs) == 0 {
		return 0, nil
	}

	// Create MMap store
	mmapStore, err := store.OpenMMapVectorStore(outputDir, 768)
	if err != nil {
		return 0, fmt.Errorf("failed to create MMap vector store: %w", err)
	}
	defer func() { _ = mmapStore.Close() }()

	// Write all vectors
	mmapStore.BeginWrite()
	for i, chunkID := range chunkIDs {
		if embeddings[i] != nil {
			mmapStore.WriteVector(chunkID, embeddings[i])
		}
	}

	if err := mmapStore.CommitWrite(); err != nil {
		return 0, fmt.Errorf("failed to commit MMap vectors: %w", err)
	}

	return mmapStore.VectorCount(), nil
}

// ExportVectorsToTQ exports all chunk vector embeddings to a compact TQ-MSE store.
func (idx *Indexer) ExportVectorsToTQ(ctx context.Context, outputDir string) (int, error) {
	exporter, ok := idx.store.(store.VectorExporter)
	if !ok {
		return 0, fmt.Errorf("store does not support vector export")
	}

	chunkIDs, embeddings, err := exporter.ExportAllVectors(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to export vectors: %w", err)
	}
	if len(chunkIDs) == 0 {
		if err := store.RemoveTQVectorStore(outputDir); err != nil {
			return 0, fmt.Errorf("remove stale TQ-MSE vector store: %w", err)
		}
		return 0, nil
	}

	return store.BuildTQVectorStore(ctx, outputDir, chunkIDs, embeddings, store.TQVectorBuildOptions{
		Dims: colbertEmbeddingDims(),
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	})
}

// ExportFileVectorsToTQ exports all file-level embeddings to a compact TQ-MSE store.
func (idx *Indexer) ExportFileVectorsToTQ(ctx context.Context, outputDir string) (int, error) {
	exporter, ok := idx.store.(store.FileVectorExporter)
	if !ok {
		if err := store.RemoveTQFileVectorStore(outputDir); err != nil {
			return 0, fmt.Errorf("remove stale file TQ-MSE vector store: %w", err)
		}
		return 0, nil
	}

	filePaths, embeddings, err := exporter.ExportFileEmbeddings(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to export file vectors: %w", err)
	}
	if len(filePaths) == 0 {
		if err := store.RemoveTQFileVectorStore(outputDir); err != nil {
			return 0, fmt.Errorf("remove stale file TQ-MSE vector store: %w", err)
		}
		return 0, nil
	}

	return store.BuildTQFileVectorStore(ctx, outputDir, filePaths, embeddings, store.TQVectorBuildOptions{
		Dims: colbertEmbeddingDims(),
		Bits: colbertTQMSEBits,
		Seed: colbertTQMSESeed,
	})
}

func (idx *Indexer) useAdaptiveColBERTSegments() bool {
	return idx.indexCfg != nil && idx.indexCfg.AdaptiveColBERTSegments
}

func (idx *Indexer) buildColBERTChunkSegments(ctx context.Context, chunks []store.ChunkInfo) (map[string][]store.ColBERTSegment, error) {
	floatSegments, err := idx.buildFloat32ColBERTChunkSegments(ctx, chunks)
	if err != nil {
		return nil, err
	}

	if idx.colbertCodec == store.ColBERTCodecTQMSE && idx.colbertTQMSE != nil {
		return encodeTQMSEChunkBatch(floatSegments, idx.colbertTQMSE, runtime.GOMAXPROCS(0)), nil
	}

	chunkSegments := make(map[string][]store.ColBERTSegment, len(floatSegments))
	for chunkID, segments := range floatSegments {
		chunkSegments[chunkID] = convertColBERTSegmentsForCodec(chunkID, segments, idx.colbertCodec, idx.colbertPQ, idx.colbertTQMSE)
	}
	return chunkSegments, nil
}

func encodeTQMSEChunkBatch(chunks map[string][]store.ColBERTSegment, tq *util.TQMSEQuantizer, workers int) map[string][]store.ColBERTSegment {
	type job struct {
		chunkID  string
		segments []store.ColBERTSegment
	}
	type result struct {
		chunkID  string
		segments []store.ColBERTSegment
	}

	workers = min(max(workers, 1), len(chunks))
	if workers == 0 {
		return map[string][]store.ColBERTSegment{}
	}
	encoded := make(map[string][]store.ColBERTSegment, len(chunks))
	if workers == 1 {
		encoder := tq.NewEncoder()
		for chunkID, segments := range chunks {
			converted, _ := convertColBERTSegmentsForCodecWithFallback(chunkID, segments, store.ColBERTCodecTQMSE, nil, tq, true, encoder)
			encoded[chunkID] = converted
		}
		return encoded
	}

	jobs := make(chan job, len(chunks))
	results := make(chan result, len(chunks))
	for chunkID, segments := range chunks {
		jobs <- job{chunkID: chunkID, segments: segments}
	}
	close(jobs)

	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			encoder := tq.NewEncoder()
			for job := range jobs {
				segments, _ := convertColBERTSegmentsForCodecWithFallback(job.chunkID, job.segments, store.ColBERTCodecTQMSE, nil, tq, true, encoder)
				results <- result{chunkID: job.chunkID, segments: segments}
			}
		})
	}
	wg.Wait()
	close(results)

	for result := range results {
		encoded[result.chunkID] = result.segments
	}
	return encoded
}

func (idx *Indexer) buildFloat32ColBERTChunkSegments(ctx context.Context, chunks []store.ChunkInfo) (map[string][]store.ColBERTSegment, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	chunkTexts := make([][]string, len(chunks))
	var allSegmentTexts []string

	for i, chunk := range chunks {
		combined := util.CombineDescriptionContent(chunk.Content, chunk.Description)
		segments := decomposeDocumentForColBERT(combined, idx.useAdaptiveColBERTSegments())
		chunkTexts[i] = segments
		allSegmentTexts = append(allSegmentTexts, segments...)
	}

	if len(allSegmentTexts) == 0 {
		return map[string][]store.ColBERTSegment{}, nil
	}

	embeddings, err := idx.embedBatchWithRetry(ctx, allSegmentTexts, 3)
	if err != nil {
		return nil, err
	}

	for i := range embeddings {
		embeddings[i] = util.NormalizeVectorCopy(embeddings[i])
	}

	chunkSegments := make(map[string][]store.ColBERTSegment, len(chunks))
	segIdx := 0
	adaptive := idx.useAdaptiveColBERTSegments()

	for i, chunk := range chunks {
		segmentTexts := chunkTexts[i]
		count := len(segmentTexts)
		if count == 0 {
			continue
		}

		chunkSegments[chunk.ID] = buildFloat32ColBERTSegments(
			segmentTexts,
			embeddings[segIdx:segIdx+count],
			adaptive,
		)
		segIdx += count
	}

	return chunkSegments, nil
}

func buildStoredColBERTSegments(chunkID string, segmentTexts []string, embeddings [][]float32, adaptive bool, codec store.ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) []store.ColBERTSegment {
	segments := buildFloat32ColBERTSegments(segmentTexts, embeddings, adaptive)
	return convertColBERTSegmentsForCodec(chunkID, segments, codec, pq, tq)
}

func buildFloat32ColBERTSegments(segmentTexts []string, embeddings [][]float32, adaptive bool) []store.ColBERTSegment {
	if len(segmentTexts) == 0 {
		return nil
	}

	segments := make([]store.ColBERTSegment, len(segmentTexts))
	for i := range segmentTexts {
		segments[i] = store.ColBERTSegment{
			SegmentIdx: i,
			Text:       segmentTexts[i],
			Embedding:  embeddings[i],
		}
	}

	if adaptive {
		target := searchpkg.AdaptiveSegmentBudgetFromRawCount(len(segmentTexts))
		if len(segments) > target {
			pooler := store.NewSegmentPooler(target, adaptiveColBERTPoolMinSim)
			segments = pooler.PoolAndMerge(segments)
			for i := range segments {
				segments[i].SegmentIdx = i
			}
		}
	}
	return segments
}

func convertColBERTSegmentsForCodec(chunkID string, segments []store.ColBERTSegment, codec store.ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) []store.ColBERTSegment {
	converted, _ := convertColBERTSegmentsForCodecWithFallback(chunkID, segments, codec, pq, tq, true, nil)
	return converted
}

func convertColBERTSegmentsForCodecStrict(chunkID string, segments []store.ColBERTSegment, codec store.ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer) ([]store.ColBERTSegment, error) {
	return convertColBERTSegmentsForCodecWithFallback(chunkID, segments, codec, pq, tq, false, nil)
}

func convertColBERTSegmentsForCodecWithFallback(chunkID string, segments []store.ColBERTSegment, codec store.ColBERTCodec, pq *util.ProductQuantizer, tq *util.TQMSEQuantizer, allowFallback bool, tqEncoder *util.TQMSEEncoder) ([]store.ColBERTSegment, error) {
	if codec == store.ColBERTCodecPQ6 && pq != nil {
		encoded, err := encodeSegmentsToPQ(chunkID, segments, pq)
		if err != nil {
			if !allowFallback {
				return nil, err
			}
			fmt.Fprintf(os.Stderr, "Warning: ColBERT PQ encode failed for chunk %s, falling back to int8: %v\n", chunkID, err)
		} else {
			return encoded, nil
		}
	}
	if codec == store.ColBERTCodecTQMSE && tq != nil {
		encoded, err := encodeSegmentsToTQMSEWithEncoder(chunkID, segments, tq, tqEncoder)
		if err != nil {
			if !allowFallback {
				return nil, err
			}
			fmt.Fprintf(os.Stderr, "Warning: ColBERT TQ-MSE encode failed for chunk %s, falling back to int8: %v\n", chunkID, err)
		} else {
			return encoded, nil
		}
	}

	for i := range segments {
		if segments[i].Embedding != nil {
			quantized, scale, min := util.QuantizeInt8(segments[i].Embedding)
			segments[i].Embedding = nil
			segments[i].EmbeddingInt8 = quantized
			segments[i].QuantScale = scale
			segments[i].QuantMin = min
		}
	}
	return segments, nil
}

func encodeSegmentsToTQMSE(chunkID string, segments []store.ColBERTSegment, tq *util.TQMSEQuantizer) ([]store.ColBERTSegment, error) {
	return encodeSegmentsToTQMSEWithEncoder(chunkID, segments, tq, nil)
}

func encodeSegmentsToTQMSEWithEncoder(chunkID string, segments []store.ColBERTSegment, tq *util.TQMSEQuantizer, encoder *util.TQMSEEncoder) ([]store.ColBERTSegment, error) {
	if tq == nil {
		return nil, fmt.Errorf("nil tqmse quantizer")
	}
	encoded := make([]store.ColBERTSegment, len(segments))
	codeSize := tq.CodeSize()
	for i, seg := range segments {
		if len(seg.TQCodes) == codeSize {
			encoded[i] = store.ColBERTSegment{
				SegmentIdx: seg.SegmentIdx,
				Text:       seg.Text,
				TQCodes:    append([]byte(nil), seg.TQCodes...),
			}
			continue
		}
		emb := seg.Embedding
		if emb == nil && len(seg.EmbeddingInt8) > 0 {
			emb = util.DequantizeInt8(seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
			emb = util.NormalizeVector(emb)
		}
		if emb == nil {
			if len(seg.TQCodes) > 0 {
				return nil, fmt.Errorf("segment %d in chunk %s has malformed TQ-MSE code length %d (expected %d)", seg.SegmentIdx, chunkID, len(seg.TQCodes), codeSize)
			}
			return nil, fmt.Errorf("segment %d in chunk %s is missing an encodable embedding", seg.SegmentIdx, chunkID)
		}
		if encoder == nil {
			encoder = tq.NewEncoder()
		}
		code, err := encoder.EncodeInto(util.TQMSECode{Codes: make([]byte, codeSize)}, emb)
		if err != nil {
			return nil, err
		}
		encoded[i] = store.ColBERTSegment{
			SegmentIdx: seg.SegmentIdx,
			Text:       seg.Text,
			TQCodes:    code.Codes,
		}
	}
	return encoded, nil
}

func encodeSegmentsToPQ(chunkID string, segments []store.ColBERTSegment, pq *util.ProductQuantizer) ([]store.ColBERTSegment, error) {
	encoded := make([]store.ColBERTSegment, len(segments))
	codeSize := pq.CodeSize()
	for i, seg := range segments {
		if len(seg.PQCodes) == codeSize {
			encoded[i] = store.ColBERTSegment{
				SegmentIdx: seg.SegmentIdx,
				Text:       seg.Text,
				PQCodes:    append([]byte(nil), seg.PQCodes...),
			}
			continue
		}
		emb := seg.Embedding
		if emb == nil && len(seg.EmbeddingInt8) > 0 {
			emb = util.DequantizeInt8(seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
			emb = util.NormalizeVector(emb)
		}
		if emb == nil {
			if len(seg.PQCodes) > 0 {
				return nil, fmt.Errorf("segment %d in chunk %s has malformed PQ code length %d (expected %d)", seg.SegmentIdx, chunkID, len(seg.PQCodes), codeSize)
			}
			return nil, fmt.Errorf("segment %d in chunk %s is missing an encodable embedding", seg.SegmentIdx, chunkID)
		}
		codes, err := pq.Encode(emb)
		if err != nil {
			return nil, err
		}
		encoded[i] = store.ColBERTSegment{
			SegmentIdx: seg.SegmentIdx,
			Text:       seg.Text,
			PQCodes:    codes,
		}
	}
	return encoded, nil
}

// decomposeDocumentForColBERT splits a document into meaningful segments for ColBERT.
// Adaptive mode keeps the raw decomposition and relies on embedding-aware pooling.
func decomposeDocumentForColBERT(content string, adaptive bool) []string {
	if adaptive {
		return searchpkg.DecomposeDocumentRaw(content)
	}
	return searchpkg.DecomposeDocument(content)
}

func (idx *Indexer) ensureColBERTCodecReady(ctx context.Context, segmentStore store.ColBERTSegmentStorer) error {
	if idx.colbertCodec == store.ColBERTCodecTQMSE {
		if idx.colbertTQMSE == nil {
			tq, err := newDefaultColBERTTQMSEQuantizer()
			if err != nil {
				return fmt.Errorf("create colbert tqmse: %w", err)
			}
			idx.colbertTQMSE = tq
		}
		idx.colbertPQ = nil
		return idx.persistColBERTMetadata(ctx)
	}

	if idx.colbertCodec != store.ColBERTCodecPQ6 {
		idx.colbertPQ = nil
		idx.colbertTQMSE = nil
		return idx.persistColBERTMetadata(ctx)
	}
	idx.colbertTQMSE = nil
	if idx.colbertPQ != nil {
		return nil
	}

	fmt.Println("Sampling stored ColBERT int8 segments for PQ training...")
	sample, totalVectors, err := idx.collectColBERTPQTrainingSample(ctx, segmentStore)
	if err != nil {
		return err
	}
	if totalVectors < colbertPQMinTrainingVectors {
		fmt.Printf("Skipping PQ6: only %d segment vectors available, keeping int8\n", totalVectors)
		idx.colbertCodec = store.ColBERTCodecInt8
		idx.colbertPQ = nil
		idx.colbertTQMSE = nil
		return idx.persistColBERTMetadata(ctx)
	}
	fmt.Printf("Training ColBERT PQ codebook on %d sampled vectors (%d total)...\n", len(sample), totalVectors)

	pq, err := util.NewProductQuantizer(util.PQConfig{
		Dims:       768,
		Subspaces:  6,
		Centroids:  256,
		Iterations: colbertPQTrainIterations,
	})
	if err != nil {
		return fmt.Errorf("create colbert pq: %w", err)
	}
	if err := pq.Train(sample, colbertPQTrainIterations); err != nil {
		return fmt.Errorf("train colbert pq: %w", err)
	}

	idx.colbertPQ = pq
	idx.colbertTQMSE = nil
	return nil
}

func (idx *Indexer) persistColBERTMetadata(ctx context.Context) error {
	metadataStore, ok := idx.store.(store.ColBERTMetadataStore)
	if !ok {
		return nil
	}
	return metadataStore.SaveColBERTMetadata(ctx, idx.colbertCodec, idx.colbertPQ, idx.colbertTQMSE)
}

func (idx *Indexer) collectColBERTPQTrainingSample(ctx context.Context, segmentStore store.ColBERTSegmentStorer) ([][]float32, int, error) {
	const fetchBatchSize = 64

	rng := rand.New(rand.NewSource(42))
	sample := make([][]float32, 0, colbertPQSampleSize)
	totalVectors := 0

	hasStoredSegments, err := segmentStore.HasColBERTSegments(ctx)
	if err != nil {
		return nil, 0, fmt.Errorf("check existing colbert segments: %w", err)
	}
	if hasStoredSegments {
		if storedSample, storedCount, err := collectPQTrainingSampleFromStoredSegments(ctx, segmentStore, fetchBatchSize, rng); err != nil {
			return nil, 0, err
		} else if storedCount > 0 {
			return storedSample, storedCount, nil
		}
	}

	offset := 0

	for {
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			return nil, 0, fmt.Errorf("get pq training chunks at offset %d: %w", offset, err)
		}
		if len(chunks) == 0 {
			break
		}

		var segmentTexts []string
		for _, chunk := range chunks {
			combined := util.CombineDescriptionContent(chunk.Content, chunk.Description)
			segments := decomposeDocumentForColBERT(combined, idx.useAdaptiveColBERTSegments())
			segmentTexts = append(segmentTexts, segments...)
		}
		if len(segmentTexts) == 0 {
			offset += len(chunks)
			continue
		}

		embeddings, err := idx.embedBatchWithRetry(ctx, segmentTexts, 3)
		if err != nil {
			return nil, 0, fmt.Errorf("embed pq training batch: %w", err)
		}
		for _, emb := range embeddings {
			norm := util.NormalizeVectorCopy(emb)
			totalVectors++
			if len(sample) < colbertPQSampleSize {
				sample = append(sample, norm)
				continue
			}
			replaceIdx := rng.Intn(totalVectors)
			if replaceIdx < len(sample) {
				sample[replaceIdx] = norm
			}
		}

		offset += len(chunks)
	}

	return sample, totalVectors, nil
}

func collectPQTrainingSampleFromStoredSegments(ctx context.Context, segmentStore store.ColBERTSegmentStorer, fetchBatchSize int, rng *rand.Rand) ([][]float32, int, error) {
	sample := make([][]float32, 0, colbertPQSampleSize)
	totalVectors := 0
	offset := 0

	for {
		chunks, err := segmentStore.GetChunksForColBERT(ctx, fetchBatchSize, offset)
		if err != nil {
			return nil, 0, fmt.Errorf("get stored pq training chunks at offset %d: %w", offset, err)
		}
		if len(chunks) == 0 {
			break
		}

		chunkIDs := make([]string, len(chunks))
		for i, chunk := range chunks {
			chunkIDs[i] = chunk.ID
		}
		segmentMap, err := segmentStore.GetColBERTSegmentsBatch(ctx, chunkIDs)
		if err != nil {
			return nil, 0, fmt.Errorf("load stored pq training segments at offset %d: %w", offset, err)
		}

		for _, chunkID := range chunkIDs {
			for _, seg := range segmentMap[chunkID] {
				if len(seg.EmbeddingInt8) == 0 {
					continue
				}
				norm := util.DequantizeInt8(seg.EmbeddingInt8, seg.QuantScale, seg.QuantMin)
				norm = util.NormalizeVector(norm)
				totalVectors++
				if len(sample) < colbertPQSampleSize {
					sample = append(sample, norm)
					continue
				}
				replaceIdx := rng.Intn(totalVectors)
				if replaceIdx < len(sample) {
					sample[replaceIdx] = norm
				}
			}
		}

		offset += len(chunks)
	}

	return sample, totalVectors, nil
}
