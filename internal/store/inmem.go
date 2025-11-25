package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

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

	s := &InMemStore{
		db:        db,
		dims:      getDims(),
		docsCache: make(map[string]*Document),
	}

	if err := s.init(); err != nil {
		db.Close()
		return nil, err
	}

	// Load all vectors into memory
	if err := s.loadVectors(); err != nil {
		db.Close()
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
	defer rows.Close()

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
	defer tx.Rollback()

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
	defer tx.Rollback()

	docStmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO documents (id, filepath, content, start_line, end_line, metadata)
		 VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer docStmt.Close()

	vecStmt, err := tx.PrepareContext(ctx,
		`INSERT OR REPLACE INTO vec_embeddings (embedding, doc_id) VALUES (?, ?)`)
	if err != nil {
		return err
	}
	defer vecStmt.Close()

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
}

// Search finds similar documents using in-memory L2 search.
func (s *InMemStore) Search(ctx context.Context, embedding []float32, limit int, threshold float64) ([]*Document, []float64, error) {
	s.mu.RLock()
	vectors := s.vectors
	docIDs := s.docIDs
	s.mu.RUnlock()

	if len(vectors) == 0 {
		return nil, nil, nil
	}

	// Compute L2 distances in-memory
	results := make([]searchResult, len(vectors))
	for i, vec := range vectors {
		results[i] = searchResult{
			id:       docIDs[i],
			distance: l2Distance(embedding, vec),
		}
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// Filter by threshold and limit
	var filtered []searchResult
	for _, r := range results {
		if r.distance > threshold {
			continue
		}
		filtered = append(filtered, r)
		if len(filtered) >= limit {
			break
		}
	}

	if len(filtered) == 0 {
		return nil, nil, nil
	}

	// Load documents
	placeholders := make([]string, len(filtered))
	args := make([]interface{}, len(filtered))
	for i, r := range filtered {
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
	defer rows.Close()

	docsByID := make(map[string]*Document, len(filtered))
	for rows.Next() {
		var doc Document
		var metadataStr string
		if err := rows.Scan(&doc.ID, &doc.FilePath, &doc.Content,
			&doc.StartLine, &doc.EndLine, &metadataStr); err != nil {
			return nil, nil, err
		}
		if metadataStr != "" {
			json.Unmarshal([]byte(metadataStr), &doc.Metadata)
		}
		docsByID[doc.ID] = &doc
	}

	// Reconstruct ordered results
	docs := make([]*Document, 0, len(filtered))
	distances := make([]float64, 0, len(filtered))
	for _, r := range filtered {
		if d, ok := docsByID[r.id]; ok {
			docs = append(docs, d)
			distances = append(distances, r.distance)
		}
	}

	return docs, distances, nil
}

// l2Distance computes L2 (Euclidean) distance between two vectors.
func l2Distance(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Stats returns index statistics.
func (s *InMemStore) Stats(ctx context.Context) (*Stats, error) {
	var stats Stats

	s.db.QueryRowContext(ctx,
		`SELECT COUNT(DISTINCT filepath) FROM documents`).Scan(&stats.Documents)
	s.db.QueryRowContext(ctx,
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
		rows.Scan(&id)
		ids = append(ids, id)
	}
	rows.Close()

	if len(ids) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	for _, id := range ids {
		tx.ExecContext(ctx, `DELETE FROM vec_embeddings WHERE doc_id = ?`, id)
		tx.ExecContext(ctx, `DELETE FROM documents WHERE id = ?`, id)
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
