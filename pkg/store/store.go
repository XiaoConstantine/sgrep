package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

const defaultDims = 768 // nomic-embed-text dimensions

// Storer interface for vector stores.
type Storer interface {
	Store(ctx context.Context, doc *Document) error
	StoreBatch(ctx context.Context, docs []*Document) error
	Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error)
	HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error)
	Stats(ctx context.Context) (*Stats, error)
	DeleteByPath(ctx context.Context, filepath string) error
	Close() error
}

// Document represents an indexed code chunk.
type Document struct {
	ID        string
	FilePath  string
	Content   string
	StartLine int
	EndLine   int
	Embedding []float32
	Metadata  map[string]string
	IsTest    bool // Whether this chunk is from a test file
}

// Stats holds index statistics.
type Stats struct {
	Documents int64
	Chunks    int64
	SizeBytes int64
}

// Store is the vector storage backend.
type Store struct {
	db   *sql.DB
	dims int
}

// Open opens or creates a store at the given path.
func Open(path string) (*Store, error) {
	// Ensure parent directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	sqlite_vec.Auto()

	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	s := &Store{db: db, dims: getDims()}

	if err := s.init(); err != nil {
		_ = db.Close()
		return nil, err
	}

	return s, nil
}

func getDims() int {
	if v := os.Getenv("SGREP_DIMS"); v != "" {
		var d int
		if _, err := fmt.Sscanf(v, "%d", &d); err == nil && d > 0 {
			return d
		}
	}
	return defaultDims
}

func (s *Store) init() error {
	// Performance PRAGMAs - critical for sqlite-vec performance
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",   // ~50MB cache
		"PRAGMA mmap_size=268435456", // 256MB mmap
		"PRAGMA busy_timeout=10000",  // Wait up to 10s for locks (concurrent workers)
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
			is_test INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
			rowid INTEGER PRIMARY KEY,
			embedding float[%d] distance_metric=l2,
			doc_id TEXT PARTITION KEY
		)`, s.dims),
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

	// Initialize FTS5 for hybrid search (BM25)
	if err := s.initFTS5(); err != nil {
		return fmt.Errorf("failed to init FTS5: %w", err)
	}

	return nil
}

// initFTS5 creates the FTS5 virtual table and sync triggers for BM25 search.
func (s *Store) initFTS5() error {
	// Create FTS5 virtual table
	_, err := s.db.Exec(`
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
			content,
			filepath,
			content='documents',
			content_rowid='rowid'
		)
	`)
	if err != nil {
		return fmt.Errorf("create FTS5 table: %w", err)
	}

	// Create triggers to keep FTS5 in sync with documents table
	triggers := []string{
		// After insert trigger
		`CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
			INSERT INTO documents_fts(rowid, content, filepath)
			VALUES (NEW.rowid, NEW.content, NEW.filepath);
		END`,
		// After delete trigger
		`CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
			INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
			VALUES ('delete', OLD.rowid, OLD.content, OLD.filepath);
		END`,
		// After update trigger
		`CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
			INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
			VALUES ('delete', OLD.rowid, OLD.content, OLD.filepath);
			INSERT INTO documents_fts(rowid, content, filepath)
			VALUES (NEW.rowid, NEW.content, NEW.filepath);
		END`,
	}

	for _, t := range triggers {
		if _, err := s.db.Exec(t); err != nil {
			// Ignore "already exists" errors for triggers
			if !strings.Contains(err.Error(), "already exists") {
				return fmt.Errorf("create trigger: %w", err)
			}
		}
	}

	return nil
}

// EnsureFTS5 checks if FTS5 table exists and populates it from existing documents.
// Call this on store open to handle migration of existing indexes.
func (s *Store) EnsureFTS5() error {
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE name='documents_fts'`).Scan(&count)
	if err != nil {
		return err
	}

	if count == 0 {
		// FTS5 table doesn't exist, init it
		if err := s.initFTS5(); err != nil {
			return err
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

// Store saves a document with its embedding.
func (s *Store) Store(ctx context.Context, doc *Document) error {
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

	// Insert document
	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata, is_test)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata), isTest)
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}

	// Insert embedding
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

	return tx.Commit()
}

// StoreBatch saves multiple documents in a single transaction.
func (s *Store) StoreBatch(ctx context.Context, docs []*Document) error {
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

	vecStmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = vecStmt.Close() }()

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

		blob, err := sqlite_vec.SerializeFloat32(doc.Embedding)
		if err != nil {
			return fmt.Errorf("failed to serialize embedding: %w", err)
		}

		_, err = vecStmt.ExecContext(ctx, blob, doc.ID)
		if err != nil {
			return fmt.Errorf("failed to insert embedding: %w", err)
		}
	}

	return tx.Commit()
}

// Search finds similar documents to the query embedding.
// Uses two-phase search: KNN first, then load documents.
func (s *Store) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	blob, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return nil, nil, err
	}

	// Phase 1: Fast KNN search - only get IDs and distances
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

	// Phase 2: Load documents for matched IDs only
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
		return nil, nil, fmt.Errorf("document fetch failed: %w", err)
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

	// Reconstruct ordered results
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

// DeleteByPath removes all documents for a file path.
func (s *Store) DeleteByPath(ctx context.Context, filepath string) error {
	// Get doc IDs first
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

	return tx.Commit()
}

// Stats returns index statistics.
func (s *Store) Stats(ctx context.Context) (*Stats, error) {
	var stats Stats

	// Count unique files
	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(DISTINCT filepath) FROM documents`).Scan(&stats.Documents)

	// Count chunks
	_ = s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM documents`).Scan(&stats.Chunks)

	// Get database size
	var dbPath string
	_ = s.db.QueryRowContext(ctx, `PRAGMA database_list`).Scan(nil, nil, &dbPath)
	if info, err := os.Stat(dbPath); err == nil {
		stats.SizeBytes = info.Size()
	}

	return &stats, nil
}

// HybridSearch combines semantic (vector) search with lexical (BM25) search.
// It fetches candidates from both sources and combines scores using weights.
func (s *Store) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]*Document, []float64, error) {
	blob, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return nil, nil, err
	}

	// Fetch more candidates for re-ranking
	fetchLimit := limit * 5
	if fetchLimit < 50 {
		fetchLimit = 50
	}

	// If no query terms, fall back to semantic-only search
	if queryTerms == "" {
		return s.Search(ctx, embedding, limit, threshold)
	}

	// Hybrid query: combine vector distance with BM25 score
	// Note: Both distance and bm25() return lower-is-better values
	query := `
		WITH semantic AS (
			SELECT doc_id, distance
			FROM vec_embeddings
			WHERE embedding MATCH ?
			ORDER BY distance
			LIMIT ?
		),
		lexical AS (
			SELECT d.id as doc_id, bm25(documents_fts) AS bm25_score
			FROM documents_fts f
			JOIN documents d ON d.rowid = f.rowid
			WHERE documents_fts MATCH ?
		)
		SELECT
			d.id, d.filepath, d.content, d.start_line, d.end_line, d.metadata, d.is_test,
			s.distance AS semantic_dist,
			COALESCE(l.bm25_score, 0) AS bm25_score,
			(? * s.distance) + (? * COALESCE(l.bm25_score, 0)) AS hybrid_score
		FROM documents d
		JOIN semantic s ON d.id = s.doc_id
		LEFT JOIN lexical l ON d.id = l.doc_id
		WHERE s.distance < ?
		ORDER BY hybrid_score
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, query,
		blob, fetchLimit,       // semantic params
		queryTerms,             // lexical params
		semanticWeight, bm25Weight, // score weights
		threshold, limit,       // filter and limit
	)
	if err != nil {
		// If FTS5 query fails (e.g., syntax error), fall back to semantic-only
		if strings.Contains(err.Error(), "fts5") {
			return s.Search(ctx, embedding, limit, threshold)
		}
		return nil, nil, fmt.Errorf("hybrid search failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var docs []*Document
	var scores []float64

	for rows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		var semanticDist, bm25Score, hybridScore float64

		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr, &isTest,
			&semanticDist, &bm25Score, &hybridScore); err != nil {
			return nil, nil, err
		}

		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1

		docs = append(docs, &doc)
		scores = append(scores, hybridScore)
	}

	return docs, scores, nil
}

// Close closes the store.
func (s *Store) Close() error {
	return s.db.Close()
}
