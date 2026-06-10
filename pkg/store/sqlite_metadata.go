package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	_ "modernc.org/sqlite"
)

// SQLiteMetadataStore is a read-only metadata and FTS store.
//
// It is used with external dense-vector artifacts so search does not need to
// open libSQL or sqlite-vec vector backends just to hydrate chunks.
type SQLiteMetadataStore struct {
	db     *sql.DB
	dbPath string
}

// OpenSQLiteMetadata opens an existing index for read-only metadata/FTS access.
func OpenSQLiteMetadata(path string) (*SQLiteMetadataStore, error) {
	dsn := fmt.Sprintf("file:%s?mode=ro&immutable=1", path)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open metadata sqlite: %w", err)
	}
	s := &SQLiteMetadataStore{db: db, dbPath: path}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("ping metadata sqlite: %w", err)
	}
	return s, nil
}

func (s *SQLiteMetadataStore) Store(context.Context, *Document) error {
	return fmt.Errorf("sqlite metadata store is read-only")
}

func (s *SQLiteMetadataStore) StoreBatch(context.Context, []*Document) error {
	return fmt.Errorf("sqlite metadata store is read-only")
}

func (s *SQLiteMetadataStore) Search(context.Context, []float32, int, float64) ([]*Document, []float64, error) {
	return nil, nil, fmt.Errorf("sqlite metadata store does not provide dense vector search")
}

func (s *SQLiteMetadataStore) HybridSearch(context.Context, []float32, string, int, float64, float64, float64) ([]*Document, []float64, error) {
	return nil, nil, fmt.Errorf("sqlite metadata store does not provide dense vector search")
}

// Stats returns index statistics from metadata tables plus DB sidecar sizes.
func (s *SQLiteMetadataStore) Stats(ctx context.Context) (*Stats, error) {
	var stats Stats
	_ = s.db.QueryRowContext(ctx, `SELECT COUNT(DISTINCT filepath) FROM documents`).Scan(&stats.Documents)
	_ = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM documents`).Scan(&stats.Chunks)
	stats.SizeBytes = sqliteIndexSizeBytes(s.dbPath)
	return &stats, nil
}

func (s *SQLiteMetadataStore) DeleteByPath(context.Context, string) error {
	return fmt.Errorf("sqlite metadata store is read-only")
}

func (s *SQLiteMetadataStore) Close() error {
	return s.db.Close()
}

// GetChunksByFilePath returns document chunks for a file path from metadata tables.
func (s *SQLiteMetadataStore) GetChunksByFilePath(ctx context.Context, filePath string) ([]*Document, error) {
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
		var metadata sql.NullString
		var isTest int
		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content, &doc.StartLine, &doc.EndLine, &metadata, &isTest); err != nil {
			return nil, err
		}
		if metadata.Valid && metadata.String != "" {
			_ = json.Unmarshal([]byte(metadata.String), &doc.Metadata)
		}
		doc.IsTest = isTest == 1
		docs = append(docs, &doc)
	}
	return docs, rows.Err()
}

// LoadDocumentsByID hydrates documents by chunk ID for external vector indexes.
func (s *SQLiteMetadataStore) LoadDocumentsByID(ctx context.Context, ids []string) (map[string]*Document, error) {
	docsByID := make(map[string]*Document, len(ids))
	if len(ids) == 0 {
		return docsByID, nil
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
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var doc Document
		var metadataStr string
		var isTest int
		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content, &doc.StartLine, &doc.EndLine, &metadataStr, &isTest); err != nil {
			return nil, err
		}
		if metadataStr != "" {
			_ = json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		doc.IsTest = isTest == 1
		docsByID[doc.ID] = &doc
	}
	return docsByID, rows.Err()
}

// BM25Scores returns FTS scores keyed by chunk ID.
func (s *SQLiteMetadataStore) BM25Scores(ctx context.Context, queryTerms string) (map[string]float64, error) {
	scores := make(map[string]float64)
	results, err := s.bm25Search(ctx, queryTerms, 0)
	if err != nil {
		return nil, err
	}
	for _, result := range results {
		scores[result.ID] = result.Score
	}
	return scores, nil
}

// BM25Search returns ranked FTS candidates keyed by chunk ID.
func (s *SQLiteMetadataStore) BM25Search(ctx context.Context, queryTerms string, limit int) ([]BM25SearchResult, error) {
	return s.bm25Search(ctx, queryTerms, limit)
}

func (s *SQLiteMetadataStore) bm25Search(ctx context.Context, queryTerms string, limit int) ([]BM25SearchResult, error) {
	if queryTerms == "" {
		return nil, nil
	}
	query := `
		SELECT d.id, bm25(documents_fts) AS score
		FROM documents_fts f
		JOIN documents d ON d.rowid = f.rowid
		WHERE documents_fts MATCH ?
	`
	args := []interface{}{queryTerms}
	if limit > 0 {
		query += ` ORDER BY score ASC LIMIT ?`
		args = append(args, limit)
	}
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		if strings.Contains(err.Error(), "fts5") || strings.Contains(err.Error(), "no such table") {
			return nil, nil
		}
		return nil, fmt.Errorf("BM25 query failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var results []BM25SearchResult
	for rows.Next() {
		var result BM25SearchResult
		if err := rows.Scan(&result.ID, &result.Score); err != nil {
			continue
		}
		results = append(results, result)
	}
	return results, rows.Err()
}

func sqliteIndexSizeBytes(path string) int64 {
	var total int64
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		if info, err := os.Stat(p); err == nil {
			total += info.Size()
		}
	}
	return total
}
