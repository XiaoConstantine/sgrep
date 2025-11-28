package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/sgrep/pkg/util"
	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

// InMemStore is a hybrid store: SQLite for persistence, in-memory for search.
type InMemStore struct {
	db        *sql.DB
	dims      int
	quantize  QuantizationMode
	mu        sync.RWMutex
	vectors   [][]float32 // All embeddings in memory (always float32 for search)
	docIDs    []string    // Corresponding document IDs
	docsCache map[string]*Document

	// fzf-inspired optimizations
	slabPool   *util.SlabPool
	partitions int
}

// InMemStoreOption configures an InMemStore.
type InMemStoreOption func(*InMemStore)

// WithInMemQuantization sets the quantization mode for persistent storage.
func WithInMemQuantization(mode QuantizationMode) InMemStoreOption {
	return func(s *InMemStore) {
		s.quantize = mode
	}
}

// OpenInMem opens a store with in-memory search capability.
func OpenInMem(path string, opts ...InMemStoreOption) (*InMemStore, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	sqlite_vec.Auto()

	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// fzf pattern: min(8 * NumCPU, 32) partitions
	numCPU := runtime.NumCPU()
	partitions := numCPU * 4 // Less aggressive than fzf since LLM is bottleneck
	if partitions > 16 {
		partitions = 16
	}
	if partitions < 2 {
		partitions = 2
	}

	dims := getDims()
	s := &InMemStore{
		db:         db,
		dims:       dims,
		quantize:   QuantizeNone,
		docsCache:  make(map[string]*Document),
		partitions: partitions,
		slabPool:   util.NewSlabPool(partitions, 10000, dims), // 10k docs per partition max
	}

	for _, opt := range opts {
		opt(s)
	}

	if err := s.init(); err != nil {
		_ = db.Close()
		return nil, err
	}

	// Load all vectors into memory
	if err := s.loadVectors(); err != nil {
		_ = db.Close()
		return nil, err
	}

	return s, nil
}

func (s *InMemStore) init() error {
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",
	}
	for _, p := range pragmas {
		if _, err := s.db.Exec(p); err != nil {
			return fmt.Errorf("failed to set pragma: %w", err)
		}
	}

	// Determine vector column type based on quantization mode
	vecColDef := s.vectorColumnDef()

	queries := []string{
		`CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			filepath TEXT NOT NULL,
			content TEXT NOT NULL,
			start_line INTEGER NOT NULL,
			end_line INTEGER NOT NULL,
			metadata TEXT,
			is_test INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
			rowid INTEGER PRIMARY KEY,
			%s,
			doc_id TEXT PARTITION KEY
		)`, vecColDef),
		`CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath)`,
		`CREATE INDEX IF NOT EXISTS idx_documents_is_test ON documents(is_test)`,
	}

	for _, q := range queries {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("failed to init schema: %w", err)
		}
	}

	return nil
}

// vectorColumnDef returns the vec0 column definition based on quantization mode.
func (s *InMemStore) vectorColumnDef() string {
	switch s.quantize {
	case QuantizeInt8:
		return fmt.Sprintf("embedding int8[%d] distance_metric=l2", s.dims)
	case QuantizeBinary:
		return fmt.Sprintf("embedding bit[%d] distance_metric=hamming", s.dims)
	default:
		return fmt.Sprintf("embedding float[%d] distance_metric=l2", s.dims)
	}
}

func (s *InMemStore) loadVectors() error {
	rows, err := s.db.Query(`SELECT doc_id, embedding FROM vec_embeddings`)
	if err != nil {
		return err
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var docID string
		var blob []byte
		if err := rows.Scan(&docID, &blob); err != nil {
			return err
		}

		vec := s.deserializeToFloat32(blob)
		if vec == nil {
			continue
		}

		s.docIDs = append(s.docIDs, docID)
		s.vectors = append(s.vectors, vec)
	}

	return nil
}

// deserializeToFloat32 converts stored bytes back to float32 based on quantization mode.
func (s *InMemStore) deserializeToFloat32(blob []byte) []float32 {
	switch s.quantize {
	case QuantizeInt8:
		return deserializeInt8ToFloat32(blob)
	case QuantizeBinary:
		return deserializeBinaryToFloat32(blob, s.dims)
	default:
		return deserializeFloat32(blob)
	}
}

// deserializeFloat32 converts raw float32 bytes back to float32 slice.
func deserializeFloat32(blob []byte) []float32 {
	if len(blob)%4 != 0 {
		return nil
	}
	n := len(blob) / 4
	vec := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(blob[i*4]) | uint32(blob[i*4+1])<<8 | uint32(blob[i*4+2])<<16 | uint32(blob[i*4+3])<<24
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

// deserializeInt8ToFloat32 converts int8 bytes to float32 (scale from [-128,127] to [-1,1]).
func deserializeInt8ToFloat32(blob []byte) []float32 {
	vec := make([]float32, len(blob))
	for i, b := range blob {
		vec[i] = float32(int8(b)) / 127.0
	}
	return vec
}

// deserializeBinaryToFloat32 converts bit vector to float32 (-1.0 for 0, 1.0 for 1).
func deserializeBinaryToFloat32(blob []byte, dims int) []float32 {
	vec := make([]float32, dims)
	for i := 0; i < dims; i++ {
		byteIdx := i / 8
		bitIdx := uint(7 - (i % 8))
		if byteIdx < len(blob) && (blob[byteIdx]&(1<<bitIdx)) != 0 {
			vec[i] = 1.0
		} else {
			vec[i] = -1.0
		}
	}
	return vec
}

// Store saves a document with its embedding.
func (s *InMemStore) Store(ctx context.Context, doc *Document) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	metadata, _ := json.Marshal(doc.Metadata)
	isTest := 0
	if doc.IsTest {
		isTest = 1
	}

	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest)
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}

	blob, err := s.serializeEmbedding(doc.Embedding)
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	vecSQL := s.vectorInsertSQL()
	_, err = tx.ExecContext(ctx, vecSQL, blob, doc.ID)
	if err != nil {
		return fmt.Errorf("failed to insert embedding: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// Update in-memory cache
	s.mu.Lock()
	s.docIDs = append(s.docIDs, doc.ID)
	s.vectors = append(s.vectors, doc.Embedding)
	s.mu.Unlock()

	return nil
}

// serializeEmbedding converts a float32 embedding to bytes based on quantization mode.
func (s *InMemStore) serializeEmbedding(embedding []float32) ([]byte, error) {
	switch s.quantize {
	case QuantizeInt8:
		int8Vec := QuantizeToInt8Unit(embedding)
		return SerializeInt8(int8Vec), nil
	case QuantizeBinary:
		return QuantizeToBinary(embedding), nil
	default:
		return sqlite_vec.SerializeFloat32(embedding)
	}
}

// vectorInsertSQL returns the INSERT statement with appropriate vec wrapper.
func (s *InMemStore) vectorInsertSQL() string {
	switch s.quantize {
	case QuantizeInt8:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (vec_int8(?), ?)`
	case QuantizeBinary:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (vec_bit(?), ?)`
	default:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`
	}
}

// StoreBatch saves multiple documents.
func (s *InMemStore) StoreBatch(ctx context.Context, docs []*Document) error {
	if len(docs) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	docStmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = docStmt.Close() }()

	// Use appropriate SQL wrapper for vector type
	vecSQL := s.vectorInsertSQL()
	vecStmt, err := tx.PrepareContext(ctx, vecSQL)
	if err != nil {
		return err
	}
	defer func() { _ = vecStmt.Close() }()

	newIDs := make([]string, 0, len(docs))
	newVecs := make([][]float32, 0, len(docs))

	for _, doc := range docs {
		metadata, _ := json.Marshal(doc.Metadata)
		isTest := 0
		if doc.IsTest {
			isTest = 1
		}

		_, err = docStmt.ExecContext(ctx,
			doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest)
		if err != nil {
			return fmt.Errorf("failed to insert document %s: %w", doc.ID, err)
		}

		blob, err := s.serializeEmbedding(doc.Embedding)
		if err != nil {
			return fmt.Errorf("failed to serialize embedding: %w", err)
		}

		_, err = vecStmt.ExecContext(ctx, blob, doc.ID)
		if err != nil {
			return fmt.Errorf("failed to insert embedding: %w", err)
		}

		newIDs = append(newIDs, doc.ID)
		newVecs = append(newVecs, doc.Embedding)
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// Update in-memory cache
	s.mu.Lock()
	s.docIDs = append(s.docIDs, newIDs...)
	s.vectors = append(s.vectors, newVecs...)
	s.mu.Unlock()

	return nil
}

type searchResult struct {
	id       string
	distance float64
	idx      int
}

// partialResult holds results from one partition.
type partialResult struct {
	partition int
	results   []searchResult
}

// Search finds similar documents using parallel partitioned in-memory L2 search.
// Uses fzf-inspired patterns: pre-allocated slabs, partitioned parallelism.
func (s *InMemStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	s.mu.RLock()
	vectors := s.vectors
	docIDs := s.docIDs
	s.mu.RUnlock()

	n := len(vectors)
	if n == 0 {
		return nil, nil, nil
	}

	// For small datasets, use simple sequential search
	if n < 1000 {
		return s.searchSequential(ctx, embedding, vectors, docIDs, limit, threshold)
	}

	// Parallel partitioned search for larger datasets
	return s.searchParallel(ctx, embedding, vectors, docIDs, limit, threshold)
}

// searchSequential is the simple path for small datasets.
// Allocates fresh buffers per call for concurrent safety.
func (s *InMemStore) searchSequential(ctx context.Context, embedding []float32, vectors [][]float32, docIDs []string, limit int, threshold float64) ([]*Document, []float64, error) {
	n := len(vectors)

	// Allocate fresh buffers for concurrent safety
	distances := make([]float64, n)
	indices := make([]int, n)

	// Compute all distances
	util.L2DistanceBatch(embedding, vectors, distances)

	// Get top-k indices
	topK := util.TopKIndices(distances, indices, limit*2) // Get more than needed for threshold filter

	// Filter by threshold and collect results
	var filtered []searchResult
	for _, idx := range topK {
		if distances[idx] > threshold {
			continue
		}
		filtered = append(filtered, searchResult{
			id:       docIDs[idx],
			distance: distances[idx],
			idx:      idx,
		})
		if len(filtered) >= limit {
			break
		}
	}

	return s.loadDocuments(ctx, filtered)
}

// searchParallel uses partitioned parallelism (fzf pattern).
func (s *InMemStore) searchParallel(ctx context.Context, embedding []float32, vectors [][]float32, docIDs []string, limit int, threshold float64) ([]*Document, []float64, error) {
	n := len(vectors)
	numPartitions := s.partitions
	if numPartitions > n {
		numPartitions = n
	}

	chunkSize := (n + numPartitions - 1) / numPartitions
	resultChan := make(chan partialResult, numPartitions)

	var cancelled atomic.Bool
	var wg sync.WaitGroup

	// Launch parallel workers
	for p := 0; p < numPartitions; p++ {
		start := p * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(partition, start, end int) {
			defer wg.Done()

			if cancelled.Load() {
				return
			}

			// Get pre-allocated slab for this partition
			slab := s.slabPool.Get(partition)
			partitionSize := end - start
			distances := slab.Distances(partitionSize)

			// Compute distances for this partition
			util.L2DistanceBatch(embedding, vectors[start:end], distances)

			// Collect results under threshold
			var results []searchResult
			for i := 0; i < partitionSize; i++ {
				if distances[i] <= threshold {
					results = append(results, searchResult{
						id:       docIDs[start+i],
						distance: distances[i],
						idx:      start + i,
					})
				}
			}

			resultChan <- partialResult{partition: partition, results: results}
		}(p, start, end)
	}

	// Close channel when all workers done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Merge results from all partitions
	var allResults []searchResult
	for pr := range resultChan {
		allResults = append(allResults, pr.results...)
	}

	// Sort merged results by distance
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].distance < allResults[j].distance
	})

	// Take top limit
	if len(allResults) > limit {
		allResults = allResults[:limit]
	}

	return s.loadDocuments(ctx, allResults)
}

// loadDocuments fetches documents from SQLite for the given search results.
func (s *InMemStore) loadDocuments(ctx context.Context, results []searchResult) ([]*Document, []float64, error) {
	if len(results) == 0 {
		return nil, nil, nil
	}

	placeholders := make([]string, len(results))
	args := make([]interface{}, len(results))
	for i, r := range results {
		placeholders[i] = "?"
		args[i] = r.id
	}

	query := fmt.Sprintf(`
		SELECT id, filepath, content, start_line, end_line, metadata, is_test
		FROM documents
		WHERE id IN (%s)
	`, strings.Join(placeholders, ","))

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = rows.Close() }()

	docsByID := make(map[string]*Document, len(results))
	for rows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr, &isTest); err != nil {
			return nil, nil, err
		}
		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1
		docsByID[doc.ID] = &doc
	}

	// Reconstruct ordered results
	docs := make([]*Document, 0, len(results))
	distances := make([]float64, 0, len(results))
	for _, r := range results {
		if d, ok := docsByID[r.id]; ok {
			docs = append(docs, d)
			distances = append(distances, r.distance)
		}
	}

	return docs, distances, nil
}

// Stats returns index statistics.
func (s *InMemStore) Stats(ctx context.Context) (*Stats, error) {
	var stats Stats

	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(DISTINCT filepath) FROM documents`).Scan(&stats.Documents)
	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM documents`).Scan(&stats.Chunks)

	s.mu.RLock()
	stats.Chunks = int64(len(s.vectors))
	s.mu.RUnlock()

	return &stats, nil
}

// DeleteByPath removes all documents for a file path.
func (s *InMemStore) DeleteByPath(ctx context.Context, filePath string) error {
	rows, err := s.db.QueryContext(ctx, `SELECT id FROM documents WHERE filepath = ?`, filePath)
	if err != nil {
		return err
	}

	var ids []string
	for rows.Next() {
		var id string
		_ = rows.Scan(&id)
		ids = append(ids, id)
	}
	_ = rows.Close()

	if len(ids) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	for _, id := range ids {
		_, _ = tx.ExecContext(ctx, `DELETE FROM vec_embeddings WHERE doc_id = ?`, id)
		_, _ = tx.ExecContext(ctx, `DELETE FROM documents WHERE id = ?`, id)
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// Rebuild in-memory cache (simple approach)
	return s.loadVectors()
}

// Close closes the store.
func (s *InMemStore) Close() error {
	return s.db.Close()
}

// EnsureFTS5 checks if FTS5 table exists and populates it from existing documents.
// Returns nil if FTS5 is not available (graceful degradation).
func (s *InMemStore) EnsureFTS5() error {
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE name='documents_fts'`).Scan(&count)
	if err != nil {
		return err
	}

	if count == 0 {
		// Create FTS5 virtual table
		_, err = s.db.Exec(`
			CREATE VIRTUAL TABLE documents_fts USING fts5(
				content,
				filepath,
				content='documents',
				content_rowid='rowid'
			)
		`)
		if err != nil {
			// FTS5 might not be available in some SQLite builds
			if strings.Contains(err.Error(), "no such module: fts5") {
				return nil // Gracefully skip FTS5
			}
			return fmt.Errorf("create FTS5 table: %w", err)
		}

		// Create sync triggers
		triggers := []string{
			`CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
				INSERT INTO documents_fts(rowid, content, filepath)
				VALUES (NEW.rowid, NEW.content, NEW.filepath);
			END`,
			`CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
				INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
				VALUES ('delete', OLD.rowid, OLD.content, OLD.filepath);
			END`,
			`CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
				INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
				VALUES ('delete', OLD.rowid, OLD.content, OLD.filepath);
				INSERT INTO documents_fts(rowid, content, filepath)
				VALUES (NEW.rowid, NEW.content, NEW.filepath);
			END`,
		}
		for _, t := range triggers {
			if _, err := s.db.Exec(t); err != nil {
				if !strings.Contains(err.Error(), "already exists") {
					return fmt.Errorf("create trigger: %w", err)
				}
			}
		}
	}

	// Check if FTS5 has data
	var ftsCount int
	err = s.db.QueryRow(`SELECT COUNT(*) FROM documents_fts`).Scan(&ftsCount)
	if err != nil {
		return err
	}

	var docCount int
	err = s.db.QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&docCount)
	if err != nil {
		return err
	}

	// If documents exist but FTS5 is empty, populate it
	if docCount > 0 && ftsCount == 0 {
		_, err = s.db.Exec(`
			INSERT INTO documents_fts(rowid, content, filepath)
			SELECT rowid, content, filepath FROM documents
		`)
		if err != nil {
			return fmt.Errorf("populate FTS5: %w", err)
		}
	}

	return nil
}

// HybridSearch combines in-memory vector search with FTS5 BM25 for hybrid ranking.
func (s *InMemStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error) {
	// If no query terms, fall back to semantic-only search
	if queryTerms == "" {
		return s.Search(ctx, embedding, limit, threshold)
	}

	// Step 1: Get semantic results from in-memory search
	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}

	s.mu.RLock()
	vectors := s.vectors
	docIDs := s.docIDs
	s.mu.RUnlock()

	n := len(vectors)
	if n == 0 {
		return nil, nil, nil
	}

	// Get semantic candidates
	var semanticCandidates []searchResult
	if n < 1000 {
		// Sequential for small datasets
		distances := make([]float64, n)
		util.L2DistanceBatch(embedding, vectors, distances)

		for i := 0; i < n; i++ {
			if distances[i] <= threshold {
				semanticCandidates = append(semanticCandidates, searchResult{
					id:       docIDs[i],
					distance: distances[i],
					idx:      i,
				})
			}
		}
	} else {
		// Parallel for large datasets
		docs, dists, err := s.searchParallel(ctx, embedding, vectors, docIDs, fetchLimit, threshold)
		if err != nil {
			return nil, nil, err
		}
		for i, doc := range docs {
			semanticCandidates = append(semanticCandidates, searchResult{
				id:       doc.ID,
				distance: dists[i],
			})
		}
	}

	if len(semanticCandidates) == 0 {
		return nil, nil, nil
	}

	// Sort by distance
	sort.Slice(semanticCandidates, func(i, j int) bool {
		return semanticCandidates[i].distance < semanticCandidates[j].distance
	})

	// Limit candidates
	if len(semanticCandidates) > fetchLimit {
		semanticCandidates = semanticCandidates[:fetchLimit]
	}

	// Step 2: Get BM25 scores from FTS5
	bm25Scores := make(map[string]float64)
	bm25Query := `
		SELECT d.id, bm25(documents_fts) AS score
		FROM documents_fts f
		JOIN documents d ON d.rowid = f.rowid
		WHERE documents_fts MATCH ?
	`
	rows, err := s.db.QueryContext(ctx, bm25Query, queryTerms)
	if err != nil {
		// If FTS5 query fails, fall back to semantic-only
		if strings.Contains(err.Error(), "fts5") {
			return s.Search(ctx, embedding, limit, threshold)
		}
		return nil, nil, fmt.Errorf("BM25 query failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var id string
		var score float64
		if err := rows.Scan(&id, &score); err != nil {
			continue
		}
		bm25Scores[id] = score
	}

	// Step 3: Compute hybrid scores
	type hybridResult struct {
		id          string
		hybridScore float64
	}

	var hybridResults []hybridResult
	for _, sr := range semanticCandidates {
		bm25 := bm25Scores[sr.id] // Will be 0 if not found (no BM25 match)
		hybrid := (semanticWeight * sr.distance) + (bm25Weight * bm25)
		hybridResults = append(hybridResults, hybridResult{
			id:          sr.id,
			hybridScore: hybrid,
		})
	}

	// Sort by hybrid score (lower is better)
	sort.Slice(hybridResults, func(i, j int) bool {
		return hybridResults[i].hybridScore < hybridResults[j].hybridScore
	})

	// Limit results
	if len(hybridResults) > limit {
		hybridResults = hybridResults[:limit]
	}

	// Step 4: Load documents
	if len(hybridResults) == 0 {
		return nil, nil, nil
	}

	placeholders := make([]string, len(hybridResults))
	args := make([]interface{}, len(hybridResults))
	for i, hr := range hybridResults {
		placeholders[i] = "?"
		args[i] = hr.id
	}

	query := fmt.Sprintf(`
		SELECT id, filepath, content, start_line, end_line, metadata, is_test
		FROM documents
		WHERE id IN (%s)
	`, strings.Join(placeholders, ","))

	docRows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = docRows.Close() }()

	docsByID := make(map[string]*Document, len(hybridResults))
	for docRows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		if err := docRows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr, &isTest); err != nil {
			return nil, nil, err
		}
		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1
		docsByID[doc.ID] = &doc
	}

	// Return in hybrid score order
	docs := make([]*Document, 0, len(hybridResults))
	scores := make([]float64, 0, len(hybridResults))
	for _, hr := range hybridResults {
		if d, ok := docsByID[hr.id]; ok {
			docs = append(docs, d)
			scores = append(scores, hr.hybridScore)
		}
	}

	return docs, scores, nil
}
