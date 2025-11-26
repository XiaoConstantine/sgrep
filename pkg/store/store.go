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
		"PRAGMA cache_size=-50000",    // ~50MB cache
		"PRAGMA mmap_size=268435456",  // 256MB mmap
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

	// Insert document
	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		doc.ID, doc.FilePath, doc.Content, doc.StartLine, doc.EndLine, string(metadata))
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
		SELECT id, filepath, content, start_line, end_line, metadata
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
		if err := docRows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr); err != nil {
			return nil, nil, err
		}
		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
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

// Close closes the store.
func (s *Store) Close() error {
	return s.db.Close()
}
