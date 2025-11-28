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

const (
	// vec0ChunkSize is the optimal batch size for sqlite-vec.
	// vec0 allocates chunks of 1024 vectors; batching inserts fills chunks efficiently.
	vec0ChunkSize = 1024

	// inMemoryThreshold is the max vector count for in-memory search.
	// Above this, we use sqlite-vec's native KNN to avoid memory bloat.
	inMemoryThreshold = 50000
)

// BufferedStore is a storage backend optimized for sqlite-vec's chunk architecture.
// It buffers writes to batch 1024 vectors per chunk, and uses adaptive search
// (in-memory for small repos, sqlite-vec KNN for large repos).
type BufferedStore struct {
	db       *sql.DB
	dims     int
	quantize QuantizationMode
	dbPath   string

	// Write buffer for batching vec0 inserts
	writeMu     sync.Mutex
	writeBuffer []bufferedDoc
	pendingDocs []*Document // Documents waiting for their embeddings to be flushed

	// Search mode
	searchMode  searchModeType
	vectorCount int64

	// In-memory search data (only populated for small repos)
	memMu   sync.RWMutex
	vectors [][]float32
	docIDs  []string

	// Parallel search config
	partitions int
	slabPool   *util.SlabPool
}

type searchModeType int

const (
	searchModeAuto     searchModeType = iota // Decide based on vector count
	searchModeInMemory                       // Force in-memory (small repos)
	searchModeSQLite                         // Force sqlite-vec KNN (large repos)
)

type bufferedDoc struct {
	docID     string
	embedding []float32
}

// BufferedStoreOption configures a BufferedStore.
type BufferedStoreOption func(*BufferedStore)

// WithBufferedQuantization sets the quantization mode.
func WithBufferedQuantization(mode QuantizationMode) BufferedStoreOption {
	return func(s *BufferedStore) {
		s.quantize = mode
	}
}

// WithSearchMode forces a specific search mode.
func WithSearchMode(mode searchModeType) BufferedStoreOption {
	return func(s *BufferedStore) {
		s.searchMode = mode
	}
}

// OpenBuffered opens a store with buffered writes and adaptive search.
func OpenBuffered(path string, opts ...BufferedStoreOption) (*BufferedStore, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	sqlite_vec.Auto()

	db, err := sql.Open("sqlite3", path)
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
	s := &BufferedStore{
		db:          db,
		dims:        dims,
		quantize:    QuantizeNone,
		dbPath:      path,
		searchMode:  searchModeAuto,
		writeBuffer: make([]bufferedDoc, 0, vec0ChunkSize),
		partitions:  partitions,
		slabPool:    util.NewSlabPool(partitions, 10000, dims),
	}

	for _, opt := range opts {
		opt(s)
	}

	if err := s.init(); err != nil {
		_ = db.Close()
		return nil, err
	}

	// Count existing vectors and decide search mode
	if err := s.initSearchMode(); err != nil {
		_ = db.Close()
		return nil, err
	}

	return s, nil
}

func (s *BufferedStore) init() error {
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",
		"PRAGMA mmap_size=268435456",
		"PRAGMA busy_timeout=10000",
	}
	for _, p := range pragmas {
		if _, err := s.db.Exec(p); err != nil {
			return fmt.Errorf("failed to set pragma: %w", err)
		}
	}

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
		`CREATE TABLE IF NOT EXISTS metadata (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
	}

	for _, q := range queries {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("failed to init schema: %w", err)
		}
	}

	return s.initFTS5()
}

func (s *BufferedStore) initFTS5() error {
	_, err := s.db.Exec(`
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
			content,
			filepath,
			content='documents',
			content_rowid='rowid'
		)
	`)
	if err != nil {
		// FTS5 might not be available in some SQLite builds (e.g., tests)
		// This is optional functionality for hybrid search
		if strings.Contains(err.Error(), "no such module: fts5") {
			return nil // Gracefully skip FTS5
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

func (s *BufferedStore) vectorColumnDef() string {
	switch s.quantize {
	case QuantizeInt8:
		return fmt.Sprintf("embedding int8[%d] distance_metric=l2", s.dims)
	case QuantizeBinary:
		return fmt.Sprintf("embedding bit[%d] distance_metric=hamming", s.dims)
	default:
		return fmt.Sprintf("embedding float[%d] distance_metric=l2", s.dims)
	}
}

// initSearchMode counts vectors and decides whether to use in-memory or sqlite-vec search.
func (s *BufferedStore) initSearchMode() error {
	var count int64
	err := s.db.QueryRow(`SELECT COUNT(*) FROM vec_embeddings`).Scan(&count)
	if err != nil {
		return err
	}
	atomic.StoreInt64(&s.vectorCount, count)

	// Decide search mode based on count
	if s.searchMode == searchModeAuto {
		if count <= inMemoryThreshold {
			s.searchMode = searchModeInMemory
		} else {
			s.searchMode = searchModeSQLite
		}
	}

	// Load vectors into memory if using in-memory search
	if s.searchMode == searchModeInMemory && count > 0 {
		return s.loadVectorsIntoMemory()
	}

	return nil
}

func (s *BufferedStore) loadVectorsIntoMemory() error {
	rows, err := s.db.Query(`SELECT doc_id, embedding FROM vec_embeddings`)
	if err != nil {
		return err
	}
	defer func() { _ = rows.Close() }()

	var docIDs []string
	var vectors [][]float32

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

		docIDs = append(docIDs, docID)
		vectors = append(vectors, vec)
	}

	s.memMu.Lock()
	s.docIDs = docIDs
	s.vectors = vectors
	s.memMu.Unlock()

	return nil
}

func (s *BufferedStore) deserializeToFloat32(blob []byte) []float32 {
	switch s.quantize {
	case QuantizeInt8:
		return deserializeInt8ToFloat32(blob)
	case QuantizeBinary:
		return deserializeBinaryToFloat32(blob, s.dims)
	default:
		return deserializeFloat32(blob)
	}
}

// Store saves a document. The embedding is buffered and flushed in batches.
func (s *BufferedStore) Store(ctx context.Context, doc *Document) error {
	// Store document metadata immediately
	metadata, _ := json.Marshal(doc.Metadata)
	isTest := 0
	if doc.IsTest {
		isTest = 1
	}

	_, err := s.db.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest)
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}

	// Buffer the embedding
	s.writeMu.Lock()
	s.writeBuffer = append(s.writeBuffer, bufferedDoc{
		docID:     doc.ID,
		embedding: doc.Embedding,
	})

	// Flush if buffer is full
	if len(s.writeBuffer) >= vec0ChunkSize {
		if err := s.flushBufferLocked(ctx); err != nil {
			s.writeMu.Unlock()
			return err
		}
	}
	s.writeMu.Unlock()

	return nil
}

// StoreBatch saves multiple documents with buffered embedding writes.
func (s *BufferedStore) StoreBatch(ctx context.Context, docs []*Document) error {
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
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// Buffer embeddings
	s.writeMu.Lock()
	for _, doc := range docs {
		s.writeBuffer = append(s.writeBuffer, bufferedDoc{
			docID:     doc.ID,
			embedding: doc.Embedding,
		})
	}

	// Flush full batches
	for len(s.writeBuffer) >= vec0ChunkSize {
		if err := s.flushBufferLocked(ctx); err != nil {
			s.writeMu.Unlock()
			return err
		}
	}
	s.writeMu.Unlock()

	return nil
}

// flushBufferLocked flushes up to vec0ChunkSize embeddings to sqlite-vec.
// Must be called with writeMu held.
func (s *BufferedStore) flushBufferLocked(ctx context.Context) error {
	if len(s.writeBuffer) == 0 {
		return nil
	}

	// Take up to vec0ChunkSize items
	batchSize := len(s.writeBuffer)
	if batchSize > vec0ChunkSize {
		batchSize = vec0ChunkSize
	}
	batch := s.writeBuffer[:batchSize]
	s.writeBuffer = s.writeBuffer[batchSize:]

	// Build a single INSERT with multiple rows for efficient chunk packing
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		// Put batch back on buffer
		s.writeBuffer = append(batch, s.writeBuffer...)
		return err
	}
	defer func() { _ = tx.Rollback() }()

	vecSQL := s.vectorInsertSQL()
	stmt, err := tx.PrepareContext(ctx, vecSQL)
	if err != nil {
		s.writeBuffer = append(batch, s.writeBuffer...)
		return err
	}
	defer func() { _ = stmt.Close() }()

	newDocIDs := make([]string, 0, batchSize)
	newVectors := make([][]float32, 0, batchSize)

	for _, bd := range batch {
		blob, err := s.serializeEmbedding(bd.embedding)
		if err != nil {
			continue
		}

		_, err = stmt.ExecContext(ctx, blob, bd.docID)
		if err != nil {
			continue
		}

		newDocIDs = append(newDocIDs, bd.docID)
		newVectors = append(newVectors, bd.embedding)
	}

	if err := tx.Commit(); err != nil {
		s.writeBuffer = append(batch, s.writeBuffer...)
		return err
	}

	// Update count and potentially in-memory cache
	atomic.AddInt64(&s.vectorCount, int64(len(newDocIDs)))

	if s.searchMode == searchModeInMemory {
		s.memMu.Lock()
		s.docIDs = append(s.docIDs, newDocIDs...)
		s.vectors = append(s.vectors, newVectors...)
		s.memMu.Unlock()
	}

	return nil
}

func (s *BufferedStore) serializeEmbedding(embedding []float32) ([]byte, error) {
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

func (s *BufferedStore) vectorInsertSQL() string {
	switch s.quantize {
	case QuantizeInt8:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (vec_int8(?), ?)`
	case QuantizeBinary:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (vec_bit(?), ?)`
	default:
		return `INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`
	}
}

// Flush writes any buffered embeddings to the database.
// Must be called before Close() to ensure all data is persisted.
func (s *BufferedStore) Flush(ctx context.Context) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	// Flush remaining buffer (may be less than vec0ChunkSize)
	for len(s.writeBuffer) > 0 {
		if err := s.flushBufferLocked(ctx); err != nil {
			return err
		}
	}

	return nil
}

// Search finds similar documents using adaptive search strategy.
func (s *BufferedStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	// Ensure buffer is flushed before search
	if err := s.Flush(ctx); err != nil {
		return nil, nil, err
	}

	if s.searchMode == searchModeInMemory {
		return s.searchInMemory(ctx, embedding, limit, threshold)
	}
	return s.searchSQLite(ctx, embedding, limit, threshold)
}

// searchInMemory performs brute-force L2 search on in-memory vectors.
func (s *BufferedStore) searchInMemory(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	s.memMu.RLock()
	vectors := s.vectors
	docIDs := s.docIDs
	s.memMu.RUnlock()

	n := len(vectors)
	if n == 0 {
		return nil, nil, nil
	}

	// Compute distances
	distances := make([]float64, n)
	util.L2DistanceBatch(embedding, vectors, distances)

	// Find top-k
	var results []inMemResult
	for i := 0; i < n; i++ {
		if distances[i] <= threshold {
			results = append(results, inMemResult{idx: i, dist: distances[i]})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	if len(results) > limit {
		results = results[:limit]
	}

	if len(results) == 0 {
		return nil, nil, nil
	}

	// Load documents
	return s.loadDocuments(ctx, results, docIDs)
}

// searchSQLite uses sqlite-vec's native KNN search.
func (s *BufferedStore) searchSQLite(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	blob, err := s.serializeEmbedding(embedding)
	if err != nil {
		return nil, nil, err
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT doc_id, distance
		FROM vec_embeddings
		WHERE embedding MATCH ?
		ORDER BY distance
		LIMIT ?
	`, blob, limit)
	if err != nil {
		return nil, nil, fmt.Errorf("KNN search failed: %w", err)
	}

	var ids []string
	var distances []float64
	for rows.Next() {
		var id string
		var distance float64
		if err := rows.Scan(&id, &distance); err != nil {
			_ = rows.Close()
			return nil, nil, err
		}
		if distance > threshold {
			continue
		}
		ids = append(ids, id)
		distances = append(distances, distance)
	}
	_ = rows.Close()

	if len(ids) == 0 {
		return nil, nil, nil
	}

	// Load documents
	placeholders := make([]string, len(ids))
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		placeholders[i] = "?"
		args[i] = id
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

	docsByID := make(map[string]*Document, len(ids))
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

	docs := make([]*Document, 0, len(ids))
	finalDistances := make([]float64, 0, len(ids))
	for i, id := range ids {
		if d, ok := docsByID[id]; ok {
			docs = append(docs, d)
			finalDistances = append(finalDistances, distances[i])
		}
	}

	return docs, finalDistances, nil
}

type inMemResult struct {
	idx  int
	dist float64
}

func (s *BufferedStore) loadDocuments(ctx context.Context, results []inMemResult, docIDs []string) ([]*Document, []float64, error) {
	if len(results) == 0 {
		return nil, nil, nil
	}

	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = docIDs[r.idx]
	}

	placeholders := make([]string, len(ids))
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		placeholders[i] = "?"
		args[i] = id
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

	docsByID := make(map[string]*Document, len(ids))
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
	for i, r := range results {
		if d, ok := docsByID[ids[i]]; ok {
			docs = append(docs, d)
			distances = append(distances, r.dist)
		}
	}

	return docs, distances, nil
}

// HybridSearch combines semantic search with BM25.
func (s *BufferedStore) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error) {
	if queryTerms == "" {
		return s.Search(ctx, embedding, limit, threshold)
	}

	// Ensure buffer is flushed
	if err := s.Flush(ctx); err != nil {
		return nil, nil, err
	}

	// Get semantic candidates
	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}

	var semanticResults []struct {
		id       string
		distance float64
	}

	if s.searchMode == searchModeInMemory {
		s.memMu.RLock()
		vectors := s.vectors
		docIDs := s.docIDs
		s.memMu.RUnlock()

		if len(vectors) > 0 {
			distances := make([]float64, len(vectors))
			util.L2DistanceBatch(embedding, vectors, distances)

			for i := 0; i < len(vectors); i++ {
				if distances[i] <= threshold {
					semanticResults = append(semanticResults, struct {
						id       string
						distance float64
					}{docIDs[i], distances[i]})
				}
			}
		}
	} else {
		blob, err := s.serializeEmbedding(embedding)
		if err != nil {
			return nil, nil, err
		}

		rows, err := s.db.QueryContext(ctx, `
			SELECT doc_id, distance
			FROM vec_embeddings
			WHERE embedding MATCH ?
			ORDER BY distance
			LIMIT ?
		`, blob, fetchLimit)
		if err != nil {
			return nil, nil, err
		}

		for rows.Next() {
			var id string
			var dist float64
			if err := rows.Scan(&id, &dist); err != nil {
				_ = rows.Close()
				return nil, nil, err
			}
			if dist <= threshold {
				semanticResults = append(semanticResults, struct {
					id       string
					distance float64
				}{id, dist})
			}
		}
		_ = rows.Close()
	}

	if len(semanticResults) == 0 {
		return nil, nil, nil
	}

	sort.Slice(semanticResults, func(i, j int) bool {
		return semanticResults[i].distance < semanticResults[j].distance
	})
	if len(semanticResults) > fetchLimit {
		semanticResults = semanticResults[:fetchLimit]
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
		id    string
		score float64
	}
	var hybridResults []hybridResult
	for _, sr := range semanticResults {
		bm25 := bm25Scores[sr.id]
		hybrid := (semanticWeight * sr.distance) + (bm25Weight * bm25)
		hybridResults = append(hybridResults, hybridResult{sr.id, hybrid})
	}

	sort.Slice(hybridResults, func(i, j int) bool {
		return hybridResults[i].score < hybridResults[j].score
	})
	if len(hybridResults) > limit {
		hybridResults = hybridResults[:limit]
	}

	// Load documents
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

	docs := make([]*Document, 0, len(hybridResults))
	scores := make([]float64, 0, len(hybridResults))
	for _, hr := range hybridResults {
		if d, ok := docsByID[hr.id]; ok {
			docs = append(docs, d)
			scores = append(scores, hr.score)
		}
	}

	return docs, scores, nil
}

// DeleteByPath removes all documents for a file path.
func (s *BufferedStore) DeleteByPath(ctx context.Context, filepath string) error {
	rows, err := s.db.QueryContext(ctx, `SELECT id FROM documents WHERE filepath = ?`, filepath)
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

	atomic.AddInt64(&s.vectorCount, -int64(len(ids)))

	// Remove from in-memory cache if applicable
	if s.searchMode == searchModeInMemory {
		idSet := make(map[string]bool, len(ids))
		for _, id := range ids {
			idSet[id] = true
		}

		s.memMu.Lock()
		newDocIDs := make([]string, 0, len(s.docIDs)-len(ids))
		newVectors := make([][]float32, 0, len(s.vectors)-len(ids))
		for i, docID := range s.docIDs {
			if !idSet[docID] {
				newDocIDs = append(newDocIDs, docID)
				newVectors = append(newVectors, s.vectors[i])
			}
		}
		s.docIDs = newDocIDs
		s.vectors = newVectors
		s.memMu.Unlock()
	}

	return nil
}

// Stats returns index statistics.
func (s *BufferedStore) Stats(ctx context.Context) (*Stats, error) {
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
func (s *BufferedStore) EnsureFTS5() error {
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
func (s *BufferedStore) VectorCount() int64 {
	return atomic.LoadInt64(&s.vectorCount)
}

// SearchMode returns the current search mode.
func (s *BufferedStore) SearchMode() string {
	switch s.searchMode {
	case searchModeInMemory:
		return "in-memory"
	case searchModeSQLite:
		return "sqlite-vec"
	default:
		return "auto"
	}
}

// Close flushes any pending writes and closes the database.
func (s *BufferedStore) Close() error {
	ctx := context.Background()
	if err := s.Flush(ctx); err != nil {
		_ = s.db.Close()
		return err
	}
	return s.db.Close()
}

// Helper to deserialize float32 from bytes
func deserializeFloat32Buffered(blob []byte) []float32 {
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
