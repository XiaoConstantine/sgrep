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
	mu        sync.RWMutex
	vectors   [][]float32 // All embeddings in memory
	docIDs    []string    // Corresponding document IDs
	docsCache map[string]*Document

	// fzf-inspired optimizations
	slabPool   *util.SlabPool
	partitions int
}

// OpenInMem opens a store with in-memory search capability.
func OpenInMem(path string) (*InMemStore, error) {
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
		docsCache:  make(map[string]*Document),
		partitions: partitions,
		slabPool:   util.NewSlabPool(partitions, 10000, dims), // 10k docs per partition max
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

	queries := []string{
		`CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			filepath TEXT NOT NULL,
			content TEXT NOT NULL,
			start_line INTEGER NOT NULL,
			end_line INTEGER NOT NULL,
			metadata TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
			rowid INTEGER PRIMARY KEY,
			embedding float[%d] distance_metric=l2,
			doc_id TEXT PARTITION KEY
		)`, s.dims),
		`CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath)`,
	}

	for _, q := range queries {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("failed to init schema: %w", err)
		}
	}

	return nil
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

		vec := deserializeFloat32(blob)
		if vec == nil {
			continue
		}

		s.docIDs = append(s.docIDs, docID)
		s.vectors = append(s.vectors, vec)
	}

	return nil
}

// deserializeFloat32 converts raw bytes back to float32 slice.
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

// Store saves a document with its embedding.
func (s *InMemStore) Store(ctx context.Context, doc *Document) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	metadata, _ := json.Marshal(doc.Metadata)

	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata))
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}

	blob, err := sqlite_vec.SerializeFloat32(doc.Embedding)
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`,
		blob, doc.ID)
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
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata)
		 VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = docStmt.Close() }()

	vecStmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = vecStmt.Close() }()

	newIDs := make([]string, 0, len(docs))
	newVecs := make([][]float32, 0, len(docs))

	for _, doc := range docs {
		metadata, _ := json.Marshal(doc.Metadata)

		_, err = docStmt.ExecContext(ctx,
			doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata))
		if err != nil {
			return fmt.Errorf("failed to insert document %s: %w", doc.ID, err)
		}

		blob, err := sqlite_vec.SerializeFloat32(doc.Embedding)
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
		SELECT id, filepath, content, start_line, end_line, metadata
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
		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr); err != nil {
			return nil, nil, err
		}
		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
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
