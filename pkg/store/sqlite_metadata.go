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
	if queryTerms == "" {
		return scores, nil
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT d.id, bm25(documents_fts) AS score
		FROM documents_fts f
		JOIN documents d ON d.rowid = f.rowid
		WHERE documents_fts MATCH ?
	`, queryTerms)
	if err != nil {
		if strings.Contains(err.Error(), "fts5") || strings.Contains(err.Error(), "no such table") {
			return scores, nil
		}
		return nil, fmt.Errorf("BM25 query failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var id string
		var score float64
		if err := rows.Scan(&id, &score); err != nil {
			continue
		}
		scores[id] = score
	}
	return scores, rows.Err()
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
