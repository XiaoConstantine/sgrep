//go:build !sqlite_vec
// +build !sqlite_vec

package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/sgrep/pkg/util"
	_ "github.com/tursodatabase/go-libsql"
)

// LibSQLStore is a vector store using libSQL's native DiskANN-based vector search.
// Unlike sqlite-vec which uses vec0 virtual tables, libSQL uses:
// - F32_BLOB column type for vectors
// - libsql_vector_idx() for DiskANN indexing
// - vector_top_k() for approximate nearest neighbor search
type LibSQLStore struct {
	db       *sql.DB
	dims     int
	dbPath   string
	quantize QuantizationMode

	// In-memory cache for small datasets (< inMemoryThreshold)
	memMu       sync.RWMutex
	vectors     [][]float32
	docIDs      []string
	vectorCount int64

	// Parallel search config
	partitions int
	slabPool   *util.SlabPool

	// Segment pooler for ColBERT compression (optional)
	segmentPooler *SegmentPooler
}

// LibSQLStoreOption configures a LibSQLStore.
type LibSQLStoreOption func(*LibSQLStore)

// WithLibSQLQuantization sets the quantization mode.
// Note: libSQL supports compress_neighbors option for index compression.
func WithLibSQLQuantization(mode QuantizationMode) LibSQLStoreOption {
	return func(s *LibSQLStore) {
		s.quantize = mode
	}
}

// WithSegmentPooling enables segment pooling for ColBERT compression.
// maxSegments: maximum segments to keep per chunk (default 5)
// minSim: minimum similarity threshold for merging (default 0.90)
func WithSegmentPooling(maxSegments int, minSim float64) LibSQLStoreOption {
	return func(s *LibSQLStore) {
		s.segmentPooler = NewSegmentPooler(maxSegments, minSim)
	}
}

// OpenLibSQL opens a store using libSQL's native vector support.
func OpenLibSQL(path string, opts ...LibSQLStoreOption) (*LibSQLStore, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	// libSQL uses "file:" prefix for local files
	dsn := path
	if !strings.HasPrefix(path, "file:") && !strings.HasPrefix(path, "libsql://") {
		dsn = "file:" + path
	}

	db, err := sql.Open("libsql", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	numCPU := runtime.NumCPU()
	partitions := numCPU * 4
	if partitions > 16 {
		partitions = 16
	}
	if partitions < 2 {
		partitions = 2
	}

	dims := getDims()
	s := &LibSQLStore{
		db:         db,
		dims:       dims,
		dbPath:     path,
		quantize:   QuantizeNone,
		partitions: partitions,
		slabPool:   util.NewSlabPool(partitions, 10000, dims),
	}

	for _, opt := range opts {
		opt(s)
	}

	if err := s.init(); err != nil {
		_ = db.Close()
		return nil, err
	}

	if err := s.initSearchMode(); err != nil {
		_ = db.Close()
		return nil, err
	}

	return s, nil
}

func (s *LibSQLStore) init() error {
	// libSQL returns result rows for PRAGMA queries, so use Query instead of Exec
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",
	}
	for _, p := range pragmas {
		rows, err := s.db.Query(p)
		if err != nil {
			return fmt.Errorf("failed to set pragma: %w", err)
		}
		_ = rows.Close()
	}

	// Create documents table with F32_BLOB for vectors (libSQL native type)
	queries := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
			id TEXT PRIMARY KEY,
			filepath TEXT NOT NULL,
			content TEXT NOT NULL,
			start_line INTEGER NOT NULL,
			end_line INTEGER NOT NULL,
			metadata TEXT,
			is_test INTEGER DEFAULT 0,
			embedding F32_BLOB(%d),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`, s.dims),
		`CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath)`,
		`CREATE INDEX IF NOT EXISTS idx_documents_is_test ON documents(is_test)`,

		// File-level embeddings for document-level search (meta-queries like "what does this repo do")
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS file_embeddings (
			filepath TEXT PRIMARY KEY,
			embedding F32_BLOB(%d),
			chunk_count INTEGER NOT NULL,
			total_lines INTEGER NOT NULL,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`, s.dims),

		// ColBERT segment embeddings for fast late-interaction scoring
		// Stores pre-computed segment embeddings per chunk to avoid query-time embedding generation
		// Uses int8 quantized embeddings (4x compression vs float32) with scale/min for dequantization
		`CREATE TABLE IF NOT EXISTS colbert_segments (
			chunk_id TEXT NOT NULL,
			segment_idx INTEGER NOT NULL,
			segment_text TEXT,
			embedding BLOB,
			quant_scale REAL,
			quant_min REAL,
			PRIMARY KEY (chunk_id, segment_idx)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_colbert_segments_chunk ON colbert_segments(chunk_id)`,
	}

	for _, q := range queries {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("failed to init schema: %w", err)
		}
	}

	// Create DiskANN vector index with compression based on quantization mode
	if err := s.createVectorIndex(); err != nil {
		return err
	}

	// Initialize FTS5 for hybrid search
	return s.initFTS5()
}

// createVectorIndex creates the DiskANN index with appropriate compression.
func (s *LibSQLStore) createVectorIndex() error {
	// Determine compression based on quantization mode
	// libSQL supports: float8, float16, int8, 1bit
	// Default max_neighbors is 3*sqrt(dims) ≈ 83 for 768-dim vectors
	// Higher values improve recall at slight index size cost
	var indexOpts string
	switch s.quantize {
	case QuantizeInt8:
		indexOpts = "'compress_neighbors=int8', 'max_neighbors=83'"
	case QuantizeBinary:
		indexOpts = "'compress_neighbors=float8', 'max_neighbors=40'"
	default:
		indexOpts = "'compress_neighbors=float8', 'max_neighbors=83'"
	}

	// Create index for documents table
	if err := s.createSingleVectorIndex("idx_documents_embedding", "documents", indexOpts); err != nil {
		return err
	}

	// Create index for file_embeddings table
	if err := s.createSingleVectorIndex("idx_file_embeddings_embedding", "file_embeddings", indexOpts); err != nil {
		return err
	}

	return nil
}

// createSingleVectorIndex creates a DiskANN index for a specific table.
func (s *LibSQLStore) createSingleVectorIndex(indexName, tableName, indexOpts string) error {
	// Check if index exists
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?`, indexName).Scan(&count)
	if err != nil {
		return err
	}

	if count > 0 {
		return nil // Index already exists
	}

	query := fmt.Sprintf(`CREATE INDEX %s ON %s (libsql_vector_idx(embedding, %s))`, indexName, tableName, indexOpts)
	_, err = s.db.Exec(query)
	if err != nil {
		// Vector index creation might fail if libSQL doesn't support it
		// Fall back to no index (brute force search)
		if strings.Contains(err.Error(), "libsql_vector_idx") {
			return nil // Gracefully continue without index
		}
		return fmt.Errorf("failed to create vector index %s: %w", indexName, err)
	}

	return nil
}

func (s *LibSQLStore) initFTS5() error {
	_, err := s.db.Exec(`
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
			content,
			filepath,
			content='documents',
			content_rowid='rowid'
		)
	`)
	if err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			return nil
		}
		return fmt.Errorf("create FTS5 table: %w", err)
	}

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

	return nil
}

func (s *LibSQLStore) initSearchMode() error {
	var count int64
	err := s.db.QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&count)
	if err != nil {
		return err
	}

	atomic.StoreInt64(&s.vectorCount, count)

	// Load vectors into memory for small datasets
	if count < inMemoryThreshold && count > 0 {
		return s.loadVectorsToMemory()
	}

	return nil
}

func (s *LibSQLStore) loadVectorsToMemory() error {
	rows, err := s.db.Query(`SELECT id, vector_extract(embedding) FROM documents WHERE embedding IS NOT NULL`)
	if err != nil {
		return err
	}
	defer func() { _ = rows.Close() }()

	s.memMu.Lock()
	defer s.memMu.Unlock()

	s.vectors = nil
	s.docIDs = nil

	for rows.Next() {
		var docID string
		var vecStr string
		if err := rows.Scan(&docID, &vecStr); err != nil {
			continue
		}

		vec := parseVectorString(vecStr)
		if vec == nil {
			continue
		}

		s.docIDs = append(s.docIDs, docID)
		s.vectors = append(s.vectors, vec)
	}

	return nil
}

// parseVectorString parses libSQL's vector string format "[1.0,2.0,3.0]"
func parseVectorString(s string) []float32 {
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	if s == "" {
		return nil
	}

	parts := strings.Split(s, ",")
	vec := make([]float32, len(parts))
	for i, p := range parts {
		var v float64
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%f", &v); err != nil {
			return nil
		}
		vec[i] = float32(v)
	}
	return vec
}

// formatVectorString formats a float32 slice as libSQL vector string
func formatVectorString(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%g", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// int8SliceToBytes converts []int8 to []byte for BLOB storage.
// This is a zero-copy cast since int8 and byte have the same size.
func int8SliceToBytes(s []int8) []byte {
	if len(s) == 0 {
		return nil
	}
	b := make([]byte, len(s))
	for i, v := range s {
		b[i] = byte(v)
	}
	return b
}

// bytesToInt8Slice converts []byte from BLOB storage back to []int8.
func bytesToInt8Slice(b []byte) []int8 {
	if len(b) == 0 {
		return nil
	}
	s := make([]int8, len(b))
	for i, v := range b {
		s[i] = int8(v)
	}
	return s
}

// Store saves a document with its embedding.
func (s *LibSQLStore) Store(ctx context.Context, doc *Document) error {
	metadata, _ := json.Marshal(doc.Metadata)
	isTest := 0
	if doc.IsTest {
		isTest = 1
	}

	// Normalize embedding for fast cosine distance via dot product
	normalizedEmb := util.NormalizeVectorCopy(doc.Embedding)
	vecStr := formatVectorString(normalizedEmb)

	_, err := s.db.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test, embedding)
		 VALUES (?, ?, ?, ?, ?, ?, ?, vector(?))`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest, vecStr)
	if err != nil {
		return err
	}

	atomic.AddInt64(&s.vectorCount, 1)

	// Update in-memory cache if using it (store normalized vector)
	count := atomic.LoadInt64(&s.vectorCount)
	if count < inMemoryThreshold {
		s.memMu.Lock()
		s.docIDs = append(s.docIDs, doc.ID)
		s.vectors = append(s.vectors, normalizedEmb)
		s.memMu.Unlock()
	}

	return nil
}

// StoreBatch saves multiple documents efficiently.
func (s *LibSQLStore) StoreBatch(ctx context.Context, docs []*Document) error {
	if len(docs) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test, embedding)
		 VALUES (?, ?, ?, ?, ?, ?, ?, vector(?))`)
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()

	for _, doc := range docs {
		metadata, _ := json.Marshal(doc.Metadata)
		isTest := 0
		if doc.IsTest {
			isTest = 1
		}
		// Normalize embedding for fast cosine distance via dot product
		normalizedEmb := util.NormalizeVectorCopy(doc.Embedding)
		vecStr := formatVectorString(normalizedEmb)

		_, err = stmt.ExecContext(ctx, doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest, vecStr)
		if err != nil {
			return err
		}
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	atomic.AddInt64(&s.vectorCount, int64(len(docs)))

	// Note: Skip in-memory cache reload during bulk indexing for performance.
	// The cache will be loaded on first search if needed.
	// Previously this reloaded ALL vectors after EVERY batch, causing O(n²) behavior.

	return nil
}

// Search finds similar documents using DiskANN index or in-memory search.
func (s *LibSQLStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	count := atomic.LoadInt64(&s.vectorCount)

	// Normalize query for dot product distance (stored vectors are pre-normalized)
	queryNorm := util.NormalizeVectorCopy(embedding)

	// Use in-memory search for small datasets
	if count < inMemoryThreshold && count > 0 {
		return s.searchInMemory(ctx, queryNorm, limit, threshold)
	}

	// Use libSQL's vector_top_k with DiskANN index
	return s.searchWithIndex(ctx, queryNorm, limit, threshold)
}

func (s *LibSQLStore) searchInMemory(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	s.memMu.RLock()
	vectors := s.vectors
	docIDs := s.docIDs
	s.memMu.RUnlock()

	// Lazy load vectors if cache is empty but we have data
	if len(vectors) == 0 {
		if err := s.loadVectorsToMemory(); err != nil {
			// Fall back to index search on error
			return s.searchWithIndex(ctx, embedding, limit, threshold)
		}
		s.memMu.RLock()
		vectors = s.vectors
		docIDs = s.docIDs
		s.memMu.RUnlock()
	}

	n := len(vectors)
	if n == 0 {
		return nil, nil, nil
	}

	// Fast dot product distance (both query and stored vectors are pre-normalized)
	distances := make([]float64, n)
	util.DotProductDistanceBatch(embedding, vectors, distances)

	// Collect results under threshold
	var results []searchResultItem
	for i := 0; i < n; i++ {
		if distances[i] <= threshold {
			results = append(results, searchResultItem{i, distances[i]})
		}
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	if len(results) == 0 {
		return nil, nil, nil
	}

	// Load documents
	return s.loadDocuments(ctx, docIDs, results)
}

func (s *LibSQLStore) searchWithIndex(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	vecStr := formatVectorString(embedding)

	// Use vector_top_k for DiskANN-based search
	// Fetch 5× candidates to compensate for ANN approximation, then filter by threshold
	fetchLimit := limit * 5
	if fetchLimit < 30 {
		fetchLimit = 30
	}

	query := `
		SELECT d.id, d.filepath, d.content, d.start_line, d.end_line, d.metadata, d.is_test,
		       vector_distance_cos(d.embedding, vector(?)) AS distance
		FROM vector_top_k('idx_documents_embedding', vector(?), ?) AS vk
		JOIN documents d ON d.rowid = vk.id
		WHERE vector_distance_cos(d.embedding, vector(?)) <= ?
		ORDER BY distance
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, query, vecStr, vecStr, fetchLimit, vecStr, threshold, limit)
	if err != nil {
		// If vector_top_k fails (index not available), fall back to brute force
		if strings.Contains(err.Error(), "vector_top_k") || strings.Contains(err.Error(), "no such table") {
			return s.searchBruteForce(ctx, embedding, limit, threshold)
		}
		return nil, nil, fmt.Errorf("vector search failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var docs []*Document
	var distances []float64

	for rows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		var distance float64

		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr, &isTest, &distance); err != nil {
			return nil, nil, err
		}

		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1

		docs = append(docs, &doc)
		distances = append(distances, distance)
	}

	return docs, distances, nil
}

func (s *LibSQLStore) searchBruteForce(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	vecStr := formatVectorString(embedding)

	query := `
		SELECT id, filepath, content, start_line, end_line, metadata, is_test,
		       vector_distance_cos(embedding, vector(?)) AS distance
		FROM documents
		WHERE embedding IS NOT NULL AND vector_distance_cos(embedding, vector(?)) <= ?
		ORDER BY distance
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, query, vecStr, vecStr, threshold, limit)
	if err != nil {
		return nil, nil, fmt.Errorf("brute force search failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var docs []*Document
	var distances []float64

	for rows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		var distance float64

		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr, &isTest, &distance); err != nil {
			return nil, nil, err
		}

		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1

		docs = append(docs, &doc)
		distances = append(distances, distance)
	}

	return docs, distances, nil
}

type searchResultItem struct {
	idx      int
	distance float64
}

func (s *LibSQLStore) loadDocuments(ctx context.Context, docIDs []string, results []searchResultItem) ([]*Document, []float64, error) {
	if len(results) == 0 {
		return nil, nil, nil
	}

	placeholders := make([]string, len(results))
	args := make([]interface{}, len(results))
	for i, r := range results {
		placeholders[i] = "?"
		args[i] = docIDs[r.idx]
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

	docs := make([]*Document, 0, len(results))
	distances := make([]float64, 0, len(results))
	for _, r := range results {
		if d, ok := docsByID[docIDs[r.idx]]; ok {
			docs = append(docs, d)
			distances = append(distances, r.distance)
		}
	}

	return docs, distances, nil
}

// HybridSearch combines vector search with FTS5 BM25.
func (s *LibSQLStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error) {
	if queryTerms == "" {
		return s.Search(ctx, embedding, limit, threshold)
	}

	// Get semantic candidates (5× for better recall before hybrid reranking)
	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}

	semanticDocs, semanticDists, err := s.Search(ctx, embedding, fetchLimit, threshold)
	if err != nil {
		return nil, nil, err
	}

	if len(semanticDocs) == 0 {
		return nil, nil, nil
	}

	// Get BM25 scores
	bm25Scores := make(map[string]float64)
	bm25Query := `
		SELECT d.id, bm25(documents_fts) AS score
		FROM documents_fts f
		JOIN documents d ON d.rowid = f.rowid
		WHERE documents_fts MATCH ?
	`
	rows, err := s.db.QueryContext(ctx, bm25Query, queryTerms)
	if err != nil {
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

	// Compute hybrid scores
	type hybridResult struct {
		doc   *Document
		score float64
	}
	var hybridResults []hybridResult
	for i, doc := range semanticDocs {
		bm25 := bm25Scores[doc.ID]
		hybrid := (semanticWeight * semanticDists[i]) + (bm25Weight * bm25)
		hybridResults = append(hybridResults, hybridResult{doc, hybrid})
	}

	sort.Slice(hybridResults, func(i, j int) bool {
		return hybridResults[i].score < hybridResults[j].score
	})

	if len(hybridResults) > limit {
		hybridResults = hybridResults[:limit]
	}

	docs := make([]*Document, len(hybridResults))
	scores := make([]float64, len(hybridResults))
	for i, hr := range hybridResults {
		docs[i] = hr.doc
		scores[i] = hr.score
	}

	return docs, scores, nil
}

// DeleteByPath removes all documents for a file path.
func (s *LibSQLStore) DeleteByPath(ctx context.Context, filepath string) error {
	result, err := s.db.ExecContext(ctx, `DELETE FROM documents WHERE filepath = ?`, filepath)
	if err != nil {
		return err
	}

	affected, _ := result.RowsAffected()
	atomic.AddInt64(&s.vectorCount, -affected)

	// Reload in-memory cache
	count := atomic.LoadInt64(&s.vectorCount)
	if count < inMemoryThreshold && count > 0 {
		return s.loadVectorsToMemory()
	}

	return nil
}

// Stats returns index statistics.
func (s *LibSQLStore) Stats(ctx context.Context) (*Stats, error) {
	var stats Stats

	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(DISTINCT filepath) FROM documents`).Scan(&stats.Documents)

	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM documents`).Scan(&stats.Chunks)

	if info, err := os.Stat(s.dbPath); err == nil {
		stats.SizeBytes = info.Size()
	}

	return &stats, nil
}

// EnsureFTS5 populates FTS5 from existing documents if needed.
func (s *LibSQLStore) EnsureFTS5() error {
	var ftsCount int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM documents_fts`).Scan(&ftsCount)
	if err != nil {
		return err
	}

	var docCount int
	err = s.db.QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&docCount)
	if err != nil {
		return err
	}

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

// VectorCount returns the number of vectors in the store.
func (s *LibSQLStore) VectorCount() int64 {
	return atomic.LoadInt64(&s.vectorCount)
}

// ExportAllVectors returns all chunk IDs and their embeddings for MMap export.
// This implements the VectorExporter interface.
func (s *LibSQLStore) ExportAllVectors(ctx context.Context) ([]string, [][]float32, error) {
	// First try in-memory cache if available
	s.memMu.RLock()
	if len(s.vectors) > 0 && len(s.docIDs) == len(s.vectors) {
		// Make copies to avoid holding lock
		ids := make([]string, len(s.docIDs))
		vecs := make([][]float32, len(s.vectors))
		copy(ids, s.docIDs)
		for i, v := range s.vectors {
			vecs[i] = make([]float32, len(v))
			copy(vecs[i], v)
		}
		s.memMu.RUnlock()
		return ids, vecs, nil
	}
	s.memMu.RUnlock()

	// Fall back to database query
	rows, err := s.db.QueryContext(ctx, `SELECT id, vector_extract(embedding) FROM documents WHERE embedding IS NOT NULL`)
	if err != nil {
		return nil, nil, fmt.Errorf("query vectors: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var ids []string
	var vecs [][]float32

	for rows.Next() {
		var docID string
		var vecStr string
		if err := rows.Scan(&docID, &vecStr); err != nil {
			continue
		}

		vec := parseVectorString(vecStr)
		if vec == nil {
			continue
		}

		ids = append(ids, docID)
		vecs = append(vecs, vec)
	}

	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate vectors: %w", err)
	}

	return ids, vecs, nil
}

// Close closes the database.
func (s *LibSQLStore) Close() error {
	return s.db.Close()
}

// DB returns the underlying database connection for direct queries.
func (s *LibSQLStore) DB() *sql.DB {
	return s.db
}

// StoreFileEmbedding stores a file-level embedding.
func (s *LibSQLStore) StoreFileEmbedding(ctx context.Context, fe *FileEmbedding) error {
	// Normalize embedding for fast cosine distance via dot product
	normalizedEmb := util.NormalizeVectorCopy(fe.Embedding)
	vecStr := formatVectorString(normalizedEmb)

	_, err := s.db.ExecContext(ctx, `
		INSERT OR REPLACE INTO file_embeddings (filepath, embedding, chunk_count, total_lines, updated_at)
		VALUES (?, vector(?), ?, ?, CURRENT_TIMESTAMP)
	`, fe.FilePath, vecStr, fe.ChunkCount, fe.TotalLines)

	return err
}

// StoreFileEmbeddingBatch stores multiple file-level embeddings.
func (s *LibSQLStore) StoreFileEmbeddingBatch(ctx context.Context, fes []*FileEmbedding) error {
	if len(fes) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT OR REPLACE INTO file_embeddings (filepath, embedding, chunk_count, total_lines, updated_at)
		VALUES (?, vector(?), ?, ?, CURRENT_TIMESTAMP)
	`)
	if err != nil {
		return fmt.Errorf("prepare: %w", err)
	}
	defer func() { _ = stmt.Close() }()

	for _, fe := range fes {
		// Normalize embedding for fast cosine distance via dot product
		normalizedEmb := util.NormalizeVectorCopy(fe.Embedding)
		vecStr := formatVectorString(normalizedEmb)
		if _, err := stmt.ExecContext(ctx, fe.FilePath, vecStr, fe.ChunkCount, fe.TotalLines); err != nil {
			return fmt.Errorf("exec for %s: %w", fe.FilePath, err)
		}
	}

	return tx.Commit()
}

// SearchFileEmbeddings searches for files by their document-level embeddings.
// Returns file paths and their distances.
func (s *LibSQLStore) SearchFileEmbeddings(ctx context.Context, embedding []float32, limit int, threshold float64) ([]string, []float64, error) {
	// Normalize query embedding
	queryNorm := util.NormalizeVectorCopy(embedding)
	vecStr := formatVectorString(queryNorm)

	// Use vector_distance_cos for ANN search on file embeddings
	query := `
		SELECT filepath, vector_distance_cos(embedding, vector(?)) as distance
		FROM file_embeddings
		WHERE embedding IS NOT NULL
		ORDER BY distance ASC
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, query, vecStr, limit)
	if err != nil {
		// Fallback to brute force if vector_distance_cos not available
		return s.searchFileEmbeddingsBruteForce(ctx, queryNorm, limit, threshold)
	}
	defer func() { _ = rows.Close() }()

	var filePaths []string
	var distances []float64

	for rows.Next() {
		var fp string
		var dist float64
		if err := rows.Scan(&fp, &dist); err != nil {
			continue
		}
		if dist <= threshold {
			filePaths = append(filePaths, fp)
			distances = append(distances, dist)
		}
	}

	return filePaths, distances, rows.Err()
}

// searchFileEmbeddingsBruteForce does brute force search when vector functions unavailable.
func (s *LibSQLStore) searchFileEmbeddingsBruteForce(ctx context.Context, queryNorm []float32, limit int, threshold float64) ([]string, []float64, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT filepath, vector_extract(embedding) FROM file_embeddings WHERE embedding IS NOT NULL`)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = rows.Close() }()

	type result struct {
		path string
		dist float64
	}
	var results []result

	for rows.Next() {
		var fp string
		var vecJSON string
		if err := rows.Scan(&fp, &vecJSON); err != nil {
			continue
		}
		vec := parseVectorString(vecJSON)
		if vec == nil || len(vec) != len(queryNorm) {
			continue
		}
		// Use dot product for distance (vectors are normalized)
		dist := util.DotProductDistance(queryNorm, vec)
		if dist <= threshold {
			results = append(results, result{fp, dist})
		}
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	filePaths := make([]string, len(results))
	distances := make([]float64, len(results))
	for i, r := range results {
		filePaths[i] = r.path
		distances[i] = r.dist
	}

	return filePaths, distances, nil
}

// GetChunksByFilePath returns all document chunks for a given file path.
func (s *LibSQLStore) GetChunksByFilePath(ctx context.Context, filePath string) ([]*Document, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, filepath, content, start_line, end_line, metadata, is_test
		FROM documents
		WHERE filepath = ?
		ORDER BY start_line ASC
	`, filePath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var docs []*Document
	for rows.Next() {
		var doc Document
		var metadataJSON sql.NullString
		var isTest int

		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content, &doc.StartLine, &doc.EndLine, &metadataJSON, &isTest); err != nil {
			continue
		}

		doc.IsTest = isTest != 0
		if metadataJSON.Valid {
			_ = json.Unmarshal([]byte(metadataJSON.String), &doc.Metadata)
		}

		docs = append(docs, &doc)
	}

	return docs, rows.Err()
}

// DeleteFileEmbedding deletes the file-level embedding for a path.
func (s *LibSQLStore) DeleteFileEmbedding(ctx context.Context, filePath string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM file_embeddings WHERE filepath = ?`, filePath)
	return err
}

// ComputeAndStoreFileEmbeddings computes document-level embeddings by averaging chunk embeddings.
// This enables document-level search for queries like "what does this repo do".
func (s *LibSQLStore) ComputeAndStoreFileEmbeddings(ctx context.Context) (int, error) {
	// Get list of unique file paths
	rows, err := s.db.QueryContext(ctx, `SELECT DISTINCT filepath FROM documents`)
	if err != nil {
		return 0, fmt.Errorf("query file paths: %w", err)
	}

	var filePaths []string
	for rows.Next() {
		var fp string
		if err := rows.Scan(&fp); err != nil {
			continue
		}
		filePaths = append(filePaths, fp)
	}
	_ = rows.Close()

	if len(filePaths) == 0 {
		return 0, nil
	}

	// Batch file embeddings for storage
	var fileEmbeddings []*FileEmbedding
	const batchSize = 100

	for _, fp := range filePaths {
		// Get all chunk embeddings for this file
		embRows, err := s.db.QueryContext(ctx, `
			SELECT vector_extract(embedding), end_line
			FROM documents
			WHERE filepath = ? AND embedding IS NOT NULL
			ORDER BY start_line
		`, fp)
		if err != nil {
			continue
		}

		var embeddings [][]float32
		var maxLine int
		for embRows.Next() {
			var vecStr string
			var endLine int
			if err := embRows.Scan(&vecStr, &endLine); err != nil {
				continue
			}
			vec := parseVectorString(vecStr)
			if vec != nil {
				embeddings = append(embeddings, vec)
				if endLine > maxLine {
					maxLine = endLine
				}
			}
		}
		_ = embRows.Close()

		if len(embeddings) == 0 {
			continue
		}

		// Compute mean embedding
		dims := len(embeddings[0])
		meanEmb := make([]float32, dims)
		for _, emb := range embeddings {
			for i, v := range emb {
				meanEmb[i] += v
			}
		}
		scale := float32(1.0 / float64(len(embeddings)))
		for i := range meanEmb {
			meanEmb[i] *= scale
		}

		fileEmbeddings = append(fileEmbeddings, &FileEmbedding{
			FilePath:   fp,
			Embedding:  meanEmb,
			ChunkCount: len(embeddings),
			TotalLines: maxLine,
		})

		// Store in batches
		if len(fileEmbeddings) >= batchSize {
			if err := s.StoreFileEmbeddingBatch(ctx, fileEmbeddings); err != nil {
				return 0, fmt.Errorf("store file embeddings: %w", err)
			}
			fileEmbeddings = nil
		}
	}

	// Store remaining
	if len(fileEmbeddings) > 0 {
		if err := s.StoreFileEmbeddingBatch(ctx, fileEmbeddings); err != nil {
			return 0, fmt.Errorf("store file embeddings: %w", err)
		}
	}

	return len(filePaths), nil
}

// ============================================================
// ColBERT Segment Storage (ColBERTSegmentStorer interface)
// ============================================================

// StoreColBERTSegments stores pre-computed segment embeddings for a chunk.
// Uses int8 quantization for 4x storage compression.
func (s *LibSQLStore) StoreColBERTSegments(ctx context.Context, chunkID string, segments []ColBERTSegment) error {
	if len(segments) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	// Delete existing segments for this chunk
	if _, err := tx.ExecContext(ctx, `DELETE FROM colbert_segments WHERE chunk_id = ?`, chunkID); err != nil {
		return err
	}

	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO colbert_segments (chunk_id, segment_idx, segment_text, embedding, quant_scale, quant_min) VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()

	for _, seg := range segments {
		// Quantize embedding to int8
		quantized, scale, min := util.QuantizeInt8(seg.Embedding)
		// Convert []int8 to []byte for BLOB storage
		embBytes := int8SliceToBytes(quantized)
		if _, err := stmt.ExecContext(ctx, chunkID, seg.SegmentIdx, seg.Text, embBytes, scale, min); err != nil {
			return err
		}
	}

	return tx.Commit()
}

// StoreColBERTSegmentsBatch stores segments for multiple chunks efficiently.
// Uses int8 quantization for 4x storage compression.
func (s *LibSQLStore) StoreColBERTSegmentsBatch(ctx context.Context, chunkSegments map[string][]ColBERTSegment) error {
	if len(chunkSegments) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	// Prepare statements
	deleteStmt, err := tx.PrepareContext(ctx, `DELETE FROM colbert_segments WHERE chunk_id = ?`)
	if err != nil {
		return err
	}
	defer func() { _ = deleteStmt.Close() }()

	insertStmt, err := tx.PrepareContext(ctx,
		`INSERT INTO colbert_segments (chunk_id, segment_idx, segment_text, embedding, quant_scale, quant_min) VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = insertStmt.Close() }()

	for chunkID, segments := range chunkSegments {
		// Delete existing segments
		if _, err := deleteStmt.ExecContext(ctx, chunkID); err != nil {
			return err
		}

		// Apply segment pooling if configured (reduces segment count via diversity sampling)
		if s.segmentPooler != nil && len(segments) > 0 {
			// First quantize all segments so pooler can work with int8
			for i := range segments {
				if segments[i].Embedding != nil && segments[i].EmbeddingInt8 == nil {
					q, scale, min := util.QuantizeInt8(segments[i].Embedding)
					segments[i].EmbeddingInt8 = q
					segments[i].QuantScale = scale
					segments[i].QuantMin = min
				}
			}
			segments = s.segmentPooler.PoolAndMerge(segments)
		}

		// Insert new segments with quantized embeddings
		for _, seg := range segments {
			var embBytes []byte
			var scale, min float32
			if seg.EmbeddingInt8 != nil {
				embBytes = int8SliceToBytes(seg.EmbeddingInt8)
				scale = seg.QuantScale
				min = seg.QuantMin
			} else if seg.Embedding != nil {
				quantized, s, m := util.QuantizeInt8(seg.Embedding)
				embBytes = int8SliceToBytes(quantized)
				scale = s
				min = m
			}
			if _, err := insertStmt.ExecContext(ctx, chunkID, seg.SegmentIdx, seg.Text, embBytes, scale, min); err != nil {
				return err
			}
		}
	}

	return tx.Commit()
}

// GetColBERTSegments retrieves pre-computed segment embeddings for a chunk.
// Returns segments with quantized int8 embeddings for efficient MaxSim computation.
func (s *LibSQLStore) GetColBERTSegments(ctx context.Context, chunkID string) ([]ColBERTSegment, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT segment_idx, segment_text, embedding, quant_scale, quant_min FROM colbert_segments WHERE chunk_id = ? ORDER BY segment_idx`,
		chunkID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var segments []ColBERTSegment
	for rows.Next() {
		var seg ColBERTSegment
		var embBytes []byte
		if err := rows.Scan(&seg.SegmentIdx, &seg.Text, &embBytes, &seg.QuantScale, &seg.QuantMin); err != nil {
			return nil, err
		}
		seg.EmbeddingInt8 = bytesToInt8Slice(embBytes)
		segments = append(segments, seg)
	}

	return segments, rows.Err()
}

// GetColBERTSegmentsBatch retrieves segments for multiple chunks efficiently.
// Returns segments with quantized int8 embeddings for efficient MaxSim computation.
func (s *LibSQLStore) GetColBERTSegmentsBatch(ctx context.Context, chunkIDs []string) (map[string][]ColBERTSegment, error) {
	if len(chunkIDs) == 0 {
		return make(map[string][]ColBERTSegment), nil
	}

	// Build query with placeholders
	placeholders := make([]string, len(chunkIDs))
	args := make([]interface{}, len(chunkIDs))
	for i, id := range chunkIDs {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(
		`SELECT chunk_id, segment_idx, segment_text, embedding, quant_scale, quant_min
		 FROM colbert_segments
		 WHERE chunk_id IN (%s)
		 ORDER BY chunk_id, segment_idx`,
		strings.Join(placeholders, ","))

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	result := make(map[string][]ColBERTSegment)
	for rows.Next() {
		var chunkID string
		var seg ColBERTSegment
		var embBytes []byte
		if err := rows.Scan(&chunkID, &seg.SegmentIdx, &seg.Text, &embBytes, &seg.QuantScale, &seg.QuantMin); err != nil {
			return nil, err
		}
		seg.EmbeddingInt8 = bytesToInt8Slice(embBytes)
		result[chunkID] = append(result[chunkID], seg)
	}

	return result, rows.Err()
}

// DeleteColBERTSegments removes segment embeddings for a chunk.
func (s *LibSQLStore) DeleteColBERTSegments(ctx context.Context, chunkID string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM colbert_segments WHERE chunk_id = ?`, chunkID)
	return err
}

// HasColBERTSegments checks if ColBERT segments exist for any chunks.
func (s *LibSQLStore) HasColBERTSegments(ctx context.Context) (bool, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM colbert_segments LIMIT 1`).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// GetChunksForColBERT retrieves chunks in paginated batches for ColBERT preindexing.
// Uses LIMIT/OFFSET pagination with deterministic ordering by rowid.
func (s *LibSQLStore) GetChunksForColBERT(ctx context.Context, batchSize int, offset int) ([]ChunkInfo, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content FROM documents
		WHERE embedding IS NOT NULL
		ORDER BY rowid
		LIMIT ? OFFSET ?
	`, batchSize, offset)
	if err != nil {
		return nil, fmt.Errorf("query chunks: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var chunks []ChunkInfo
	for rows.Next() {
		var chunk ChunkInfo
		if err := rows.Scan(&chunk.ID, &chunk.Content); err != nil {
			return nil, fmt.Errorf("scan chunk: %w", err)
		}
		chunks = append(chunks, chunk)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate chunks: %w", err)
	}

	return chunks, nil
}

