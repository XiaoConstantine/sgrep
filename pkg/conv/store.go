package conv

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
	"time"
)

const (
	// Schema version for migrations
	schemaVersion = 2
	// Default embedding dimensions for nomic-embed-text
	defaultDims = 768
)

// Store handles conversation storage and retrieval.
type Store struct {
	db       *sql.DB
	dbPath   string
	dims     int
	mmapPath string
}

// StoreConfig configures the conversation store.
type StoreConfig struct {
	DBPath string
	Dims   int
}

// DefaultStoreConfig returns default configuration.
func DefaultStoreConfig() StoreConfig {
	homeDir, _ := os.UserHomeDir()
	return StoreConfig{
		DBPath: filepath.Join(homeDir, ".sgrep", "conversations", "conv.db"),
		Dims:   defaultDims,
	}
}

// NewStore creates a new conversation store.
func NewStore(cfg StoreConfig) (*Store, error) {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(cfg.DBPath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create store directory: %w", err)
	}

	store, err := openStore(cfg.DBPath, cfg.Dims, false)
	if err != nil {
		return nil, err
	}

	// Initialize schema
	if err := store.initSchema(); err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return store, nil
}

// OpenStore opens an existing store.
func OpenStore(dbPath string) (*Store, error) {
	return openStore(dbPath, defaultDims, false)
}

// OpenStoreReadOnly opens an existing store without running schema migrations or writer pragmas.
// The connection is set to query_only mode so read commands cannot accidentally mutate the DB.
func OpenStoreReadOnly(dbPath string) (*Store, error) {
	store, err := openStore(dbPath, defaultDims, true)
	if err != nil {
		return nil, err
	}
	if err := store.applyReadPragmas(); err != nil {
		_ = store.Close()
		return nil, err
	}
	return store, nil
}

func openStore(dbPath string, dims int, existingOnly bool) (*Store, error) {
	if dims == 0 {
		dims = defaultDims
	}

	dsn := dbPath
	if (sqliteDriverName == "libsql" || existingOnly) &&
		!strings.HasPrefix(dsn, "file:") &&
		!strings.HasPrefix(dsn, "libsql://") {
		dsn = "file:" + dsn
	}
	if existingOnly && strings.HasPrefix(dsn, "file:") {
		dsn = appendDSNParam(dsn, "mode", "rw")
	}

	db, err := sql.Open(sqliteDriverName, dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}
	db.SetMaxOpenConns(1)

	return &Store{
		db:       db,
		dbPath:   dbPath,
		dims:     dims,
		mmapPath: filepath.Join(filepath.Dir(dbPath), "embeddings.mmap"),
	}, nil
}

func appendDSNParam(dsn, key, value string) string {
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	return dsn + sep + key + "=" + value
}

// Close closes the store.
func (s *Store) Close() error {
	return s.db.Close()
}

// initSchema creates the database schema.
func (s *Store) initSchema() error {
	if err := s.applyPragmas(); err != nil {
		return err
	}

	// Execute schema statements individually for better compatibility
	statements := []string{
		// Metadata table for schema version
		`CREATE TABLE IF NOT EXISTS conv_metadata (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)`,
		// Sessions table (one row per conversation)
		`CREATE TABLE IF NOT EXISTS conv_sessions (
			id TEXT PRIMARY KEY,
			agent TEXT NOT NULL,
			agent_version TEXT,
			source_path TEXT NOT NULL,
			project_path TEXT,
			project_name TEXT,
			git_branch TEXT,
			git_commit TEXT,
			started_at DATETIME NOT NULL,
			ended_at DATETIME,
			total_turns INTEGER NOT NULL,
			total_tokens INTEGER,
			metadata TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_agent ON conv_sessions(agent)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_project ON conv_sessions(project_path)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_started ON conv_sessions(started_at)`,
		// Turns table (one row per user-assistant exchange)
		`CREATE TABLE IF NOT EXISTS conv_turns (
			id TEXT PRIMARY KEY,
			session_id TEXT NOT NULL REFERENCES conv_sessions(id),
			turn_index INTEGER NOT NULL,
			user_content TEXT NOT NULL,
			assistant_content TEXT NOT NULL,
			combined_content TEXT NOT NULL,
			timestamp DATETIME,
			has_code BOOLEAN DEFAULT FALSE,
			code_langs TEXT,
			parent_uuid TEXT,
			is_sidechain BOOLEAN DEFAULT FALSE,
			UNIQUE(session_id, turn_index)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_turns_session ON conv_turns(session_id)`,
		// Embeddings table using libSQL F32_BLOB
		`CREATE TABLE IF NOT EXISTS conv_turn_embeddings (
			turn_id TEXT PRIMARY KEY,
			embedding F32_BLOB(768)
		)`,
		// Full-text search index
		`CREATE VIRTUAL TABLE IF NOT EXISTS conv_turns_fts USING fts5(
			user_content,
			assistant_content,
			content='conv_turns',
			content_rowid='rowid',
			tokenize='porter unicode61'
		)`,
		// Triggers to keep FTS in sync
		`CREATE TRIGGER IF NOT EXISTS conv_turns_ai AFTER INSERT ON conv_turns BEGIN
			INSERT INTO conv_turns_fts(rowid, user_content, assistant_content)
			VALUES (new.rowid, new.user_content, new.assistant_content);
		END`,
		`CREATE TRIGGER IF NOT EXISTS conv_turns_ad AFTER DELETE ON conv_turns BEGIN
			INSERT INTO conv_turns_fts(conv_turns_fts, rowid, user_content, assistant_content)
			VALUES('delete', old.rowid, old.user_content, old.assistant_content);
		END`,
		`CREATE TRIGGER IF NOT EXISTS conv_turns_au AFTER UPDATE ON conv_turns BEGIN
			INSERT INTO conv_turns_fts(conv_turns_fts, rowid, user_content, assistant_content)
			VALUES('delete', old.rowid, old.user_content, old.assistant_content);
			INSERT INTO conv_turns_fts(rowid, user_content, assistant_content)
			VALUES (new.rowid, new.user_content, new.assistant_content);
		END`,
	}

	for _, stmt := range statements {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("failed to execute schema statement: %w\nStatement: %s", err, stmt[:min(100, len(stmt))])
		}
	}

	vacuumNeeded, err := s.dropLegacyVectorIndex()
	if err != nil {
		return err
	}
	if vacuumNeeded {
		if _, err := s.db.Exec(`VACUUM`); err != nil {
			return fmt.Errorf("failed to vacuum conversation store after dropping legacy vector index: %w", err)
		}
	}

	// Set schema version
	_, err = s.db.Exec(`
		INSERT OR REPLACE INTO conv_metadata (key, value)
		VALUES ('schema_version', ?)
	`, schemaVersion)

	return err
}

func (s *Store) applyPragmas() error {
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",
		"PRAGMA busy_timeout=10000",
	}

	for _, pragma := range pragmas {
		if err := s.execPragma(pragma); err != nil {
			return fmt.Errorf("failed to set pragma %q: %w", pragma, err)
		}
	}

	return nil
}

func (s *Store) applyReadPragmas() error {
	pragmas := []string{
		"PRAGMA busy_timeout=10000",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA cache_size=-50000",
		"PRAGMA query_only=ON",
	}
	for _, pragma := range pragmas {
		if err := s.execPragma(pragma); err != nil {
			return fmt.Errorf("failed to set pragma %q: %w", pragma, err)
		}
	}
	return nil
}

func (s *Store) execPragma(pragma string) error {
	if sqliteDriverName == "libsql" {
		rows, err := s.db.Query(pragma)
		if err != nil {
			return err
		}
		return rows.Close()
	}

	_, err := s.db.Exec(pragma)
	return err
}

func (s *Store) dropLegacyVectorIndex() (bool, error) {
	const legacyCountQuery = `
		SELECT COUNT(*) FROM sqlite_master
		WHERE name IN ('idx_turn_embeddings_vec', 'idx_turn_embeddings_vec_shadow', 'idx_turn_embeddings_vec_shadow_idx')
	`

	var legacyCount int
	if err := s.db.QueryRow(legacyCountQuery).Scan(&legacyCount); err != nil {
		return false, fmt.Errorf("failed to inspect legacy conversation vector index: %w", err)
	}
	if legacyCount == 0 {
		return false, nil
	}

	drops := []string{
		`DROP INDEX IF EXISTS idx_turn_embeddings_vec`,
		`DROP INDEX IF EXISTS idx_turn_embeddings_vec_shadow_idx`,
		`DROP TABLE IF EXISTS idx_turn_embeddings_vec_shadow`,
	}
	for _, stmt := range drops {
		if _, err := s.db.Exec(stmt); err != nil {
			return false, fmt.Errorf("failed to drop legacy conversation vector artifact: %w", err)
		}
	}

	return true, nil
}

// StoreSession stores a session and its turns.
func (s *Store) StoreSession(ctx context.Context, session *Session) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	// Insert session
	_, err = tx.ExecContext(ctx, `
		INSERT OR REPLACE INTO conv_sessions (
			id, agent, agent_version, source_path, project_path, project_name,
			git_branch, git_commit, started_at, ended_at, total_turns, total_tokens
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`,
		session.ID, session.Agent, session.AgentVersion, session.SourcePath,
		session.ProjectPath, session.ProjectName, session.GitBranch, session.GitCommit,
		session.StartedAt, session.EndedAt, len(session.Turns), session.TotalTokens,
	)
	if err != nil {
		return fmt.Errorf("failed to insert session: %w", err)
	}

	// Insert turns
	for _, turn := range session.Turns {
		turnID := fmt.Sprintf("%s:%d", session.ID, turn.Index)
		combinedContent := fmt.Sprintf("USER: %s\n\nASSISTANT: %s", turn.UserContent, turn.AssistContent)

		codeLangsJSON := ""
		if len(turn.CodeLangs) > 0 {
			data, _ := json.Marshal(turn.CodeLangs)
			codeLangsJSON = string(data)
		}

		_, err = tx.ExecContext(ctx, `
			INSERT OR REPLACE INTO conv_turns (
				id, session_id, turn_index, user_content, assistant_content, combined_content,
				timestamp, has_code, code_langs, parent_uuid, is_sidechain
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`,
			turnID, session.ID, turn.Index, turn.UserContent, turn.AssistContent, combinedContent,
			turn.Timestamp, turn.HasCode, codeLangsJSON, turn.ParentUUID, turn.IsSidechain,
		)
		if err != nil {
			return fmt.Errorf("failed to insert turn: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		_ = tx.Rollback()
		return err
	}
	committed = true
	return nil
}

// StoreTurnEmbedding stores an embedding for a turn.
func (s *Store) StoreTurnEmbedding(ctx context.Context, turnID string, embedding []float32) error {
	blob := float32ToBlob(embedding)
	var err error
	if sqliteDriverName == "libsql" {
		_, err = s.db.ExecContext(ctx, `
			INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding)
			VALUES (?, vector32(?))
		`, turnID, blob)
	} else {
		// For sqlite3, store raw blob
		_, err = s.db.ExecContext(ctx, `
			INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding)
			VALUES (?, ?)
		`, turnID, blob)
	}
	return err
}

// StoreTurnEmbeddingBatch stores embeddings for multiple turns.
func (s *Store) StoreTurnEmbeddingBatch(ctx context.Context, turnIDs []string, embeddings [][]float32) error {
	if len(turnIDs) != len(embeddings) {
		return fmt.Errorf("turnIDs and embeddings length mismatch")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	var stmtSQL string
	if sqliteDriverName == "libsql" {
		stmtSQL = `INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding) VALUES (?, vector32(?))`
	} else {
		stmtSQL = `INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding) VALUES (?, ?)`
	}

	stmt, err := tx.PrepareContext(ctx, stmtSQL)
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()

	for i, turnID := range turnIDs {
		blob := float32ToBlob(embeddings[i])
		_, err = stmt.ExecContext(ctx, turnID, blob)
		if err != nil {
			return fmt.Errorf("failed to store embedding for %s: %w", turnID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		_ = tx.Rollback()
		return err
	}
	committed = true
	return nil
}

// GetSession retrieves a session by ID.
func (s *Store) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT id, agent, agent_version, source_path, project_path, project_name,
		       git_branch, git_commit, started_at, ended_at, total_turns, total_tokens
		FROM conv_sessions WHERE id = ?
	`, sessionID)

	var session Session
	var endedAt sql.NullTime
	var totalTurns int // Scanned but not stored; we use len(session.Turns) instead
	err := row.Scan(
		&session.ID, &session.Agent, &session.AgentVersion, &session.SourcePath,
		&session.ProjectPath, &session.ProjectName, &session.GitBranch, &session.GitCommit,
		&session.StartedAt, &endedAt, &totalTurns, &session.TotalTokens,
	)
	if err != nil {
		return nil, err
	}
	if endedAt.Valid {
		session.EndedAt = endedAt.Time
	}
	session.Agent = NormalizeAgentType(session.Agent)

	// Get turns
	rows, err := s.db.QueryContext(ctx, `
		SELECT turn_index, user_content, assistant_content, timestamp, has_code,
		       code_langs, parent_uuid, is_sidechain
		FROM conv_turns WHERE session_id = ? ORDER BY turn_index
	`, sessionID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var turn Turn
		var timestamp sql.NullTime
		var codeLangsJSON sql.NullString
		var parentUUID sql.NullString

		err := rows.Scan(
			&turn.Index, &turn.UserContent, &turn.AssistContent, &timestamp,
			&turn.HasCode, &codeLangsJSON, &parentUUID, &turn.IsSidechain,
		)
		if err != nil {
			return nil, err
		}

		if timestamp.Valid {
			turn.Timestamp = timestamp.Time
		}
		if codeLangsJSON.Valid && codeLangsJSON.String != "" {
			_ = json.Unmarshal([]byte(codeLangsJSON.String), &turn.CodeLangs)
		}
		if parentUUID.Valid {
			turn.ParentUUID = parentUUID.String
		}

		session.Turns = append(session.Turns, turn)
	}

	return &session, rows.Err()
}

// SearchResult represents a raw search result from the store.
type SearchResult struct {
	TurnID        string
	SessionID     string
	TurnIndex     int
	Score         float64
	UserContent   string
	AssistContent string
	MatchType     string
}

// VectorSearch performs vector similarity search on turn embeddings.
// Uses manual cosine similarity since libSQL's vector_top_k may not be available.
// Returns results with Score as similarity (0-1, higher is better).
func (s *Store) VectorSearch(ctx context.Context, embedding []float32, limit int, threshold float64) ([]SearchResult, error) {
	// Get all embeddings and compute similarity manually
	// This is less efficient than native vector search but works with any SQLite
	rows, err := s.db.QueryContext(ctx, `
		SELECT e.turn_id, e.embedding, t.session_id, t.turn_index, t.user_content, t.assistant_content
		FROM conv_turn_embeddings e
		JOIN conv_turns t ON e.turn_id = t.id
	`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var results []SearchResult

	for rows.Next() {
		var r SearchResult
		var embBlob []byte
		err := rows.Scan(&r.TurnID, &embBlob, &r.SessionID, &r.TurnIndex, &r.UserContent, &r.AssistContent)
		if err != nil {
			continue
		}

		// Parse embedding from blob
		docEmb := blobToFloat32(embBlob)
		if len(docEmb) != len(embedding) {
			continue
		}

		// Calculate cosine similarity (0-1, higher is better)
		similarity := cosineSimilarity(embedding, docEmb)
		if similarity >= threshold {
			r.Score = similarity
			r.MatchType = "semantic"
			results = append(results, r)
		}
	}

	// Sort by similarity (higher is better)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return results, rows.Err()
}

// HybridSearch combines vector search with BM25 text search.
func (s *Store) HybridSearch(ctx context.Context, embedding []float32, queryTerms string, limit int, threshold float64, semanticWeight, bm25Weight float64) ([]SearchResult, error) {
	// First get vector search results
	vecResults, err := s.VectorSearch(ctx, embedding, limit*2, threshold)
	if err != nil {
		return nil, err
	}

	// Then get FTS results
	ftsQuery := `
		SELECT
			t.id,
			t.session_id,
			t.turn_index,
			t.user_content,
			t.assistant_content,
			bm25(conv_turns_fts) as bm25_score
		FROM conv_turns_fts fts
		JOIN conv_turns t ON t.rowid = fts.rowid
		WHERE conv_turns_fts MATCH ?
		ORDER BY bm25_score
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, ftsQuery, queryTerms, limit*2)
	if err != nil {
		// Fall back to vector-only if FTS fails
		return vecResults, nil
	}
	defer func() { _ = rows.Close() }()

	ftsOrder := make([]string, 0, limit*2)
	ftsSeen := make(map[string]struct{}, limit*2)
	for rows.Next() {
		var turnID, sessionID, userContent, assistContent string
		var turnIndex int
		var bm25Score float64
		if err := rows.Scan(&turnID, &sessionID, &turnIndex, &userContent, &assistContent, &bm25Score); err != nil {
			continue
		}
		_ = bm25Score
		if _, ok := ftsSeen[turnID]; ok {
			continue
		}
		ftsSeen[turnID] = struct{}{}
		ftsOrder = append(ftsOrder, turnID)
	}

	// Combine results using RRF (Reciprocal Rank Fusion)
	combined := make(map[string]*SearchResult)
	rrfScores := make(map[string]float64)
	const k = 60.0

	// Add vector results
	for i, r := range vecResults {
		combined[r.TurnID] = &r
		rrfScores[r.TurnID] = semanticWeight / (k + float64(i+1))
	}

	// Add/combine FTS results
	for rank, turnID := range ftsOrder {
		if _, exists := combined[turnID]; !exists {
			// Need to fetch full result
			var r SearchResult
			err := s.db.QueryRowContext(ctx, `
				SELECT id, session_id, turn_index, user_content, assistant_content
				FROM conv_turns WHERE id = ?
			`, turnID).Scan(&r.TurnID, &r.SessionID, &r.TurnIndex, &r.UserContent, &r.AssistContent)
			if err == nil {
				combined[turnID] = &r
			}
		}
		if combined[turnID] != nil {
			rrfScores[turnID] += bm25Weight / (k + float64(rank+1))
		}
	}

	// Sort by RRF score
	var results []SearchResult
	for turnID, result := range combined {
		result.Score = rrfScores[turnID]
		result.MatchType = "hybrid"
		results = append(results, *result)
	}

	// Sort by score descending (higher RRF is better)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// FilteredSearch performs search with filters.
func (s *Store) FilteredSearch(ctx context.Context, embedding []float32, opts SearchOptions) ([]SearchResult, error) {
	// Build WHERE clause for filters
	conditions, args := buildSessionFilters(opts, "sess")

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	// Get filtered turns with embeddings
	query := fmt.Sprintf(`
		SELECT e.turn_id, e.embedding, t.session_id, t.turn_index, t.user_content, t.assistant_content
		FROM conv_turn_embeddings e
		JOIN conv_turns t ON e.turn_id = t.id
		JOIN conv_sessions sess ON t.session_id = sess.id
		%s
	`, whereClause)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var results []SearchResult

	for rows.Next() {
		var r SearchResult
		var embBlob []byte
		err := rows.Scan(&r.TurnID, &embBlob, &r.SessionID, &r.TurnIndex, &r.UserContent, &r.AssistContent)
		if err != nil {
			continue
		}

		docEmb := blobToFloat32(embBlob)
		if len(docEmb) != len(embedding) {
			continue
		}

		// Calculate cosine similarity (0-1, higher is better)
		similarity := cosineSimilarity(embedding, docEmb)
		if similarity >= opts.Threshold {
			r.Score = similarity
			r.MatchType = "semantic"
			results = append(results, r)
		}
	}

	// Sort by similarity (higher is better)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > opts.Limit {
		results = results[:opts.Limit]
	}

	return results, rows.Err()
}

// KeywordSearch performs FTS-only conversation search for exact keyword mode.
func (s *Store) KeywordSearch(ctx context.Context, queryTerms string, opts SearchOptions) ([]SearchResult, error) {
	if strings.TrimSpace(queryTerms) == "" || opts.Limit <= 0 {
		return nil, nil
	}

	conditions, args := buildSessionFilters(opts, "sess")
	whereParts := append([]string{"conv_turns_fts MATCH ?"}, conditions...)
	queryArgs := make([]interface{}, 0, len(args)+2)
	queryArgs = append(queryArgs, queryTerms)
	queryArgs = append(queryArgs, args...)
	queryArgs = append(queryArgs, opts.Limit)

	query := fmt.Sprintf(`
		SELECT
			t.id,
			t.session_id,
			t.turn_index,
			t.user_content,
			t.assistant_content,
			bm25(conv_turns_fts) as bm25_score
		FROM conv_turns_fts fts
		JOIN conv_turns t ON t.rowid = fts.rowid
		JOIN conv_sessions sess ON t.session_id = sess.id
		WHERE %s
		ORDER BY bm25_score
		LIMIT ?
	`, strings.Join(whereParts, " AND "))

	rows, err := s.db.QueryContext(ctx, query, queryArgs...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		var bm25Score float64
		if err := rows.Scan(&r.TurnID, &r.SessionID, &r.TurnIndex, &r.UserContent, &r.AssistContent, &bm25Score); err != nil {
			return nil, err
		}
		r.Score = 1.0 / float64(len(results)+1)
		r.MatchType = "keyword"
		results = append(results, r)
	}

	return results, rows.Err()
}

func buildSessionFilters(opts SearchOptions, alias string) ([]string, []interface{}) {
	var conditions []string
	var args []interface{}

	if opts.Agent != AgentAll {
		if opts.Agent == AgentPiMono {
			conditions = append(conditions, fmt.Sprintf("(%s.agent = ? OR %s.agent = ?)", alias, alias))
			args = append(args, AgentPiMono, "pi-mono")
		} else {
			conditions = append(conditions, fmt.Sprintf("%s.agent = ?", alias))
			args = append(args, opts.Agent)
		}
	}

	if opts.Project != "" {
		conditions = append(conditions, fmt.Sprintf("(%s.project_name LIKE ? OR %s.project_path LIKE ?)", alias, alias))
		args = append(args, "%"+opts.Project+"%", "%"+opts.Project+"%")
	}

	if !opts.Since.IsZero() {
		conditions = append(conditions, fmt.Sprintf("%s.started_at >= ?", alias))
		args = append(args, opts.Since)
	}

	if !opts.Before.IsZero() {
		conditions = append(conditions, fmt.Sprintf("%s.started_at <= ?", alias))
		args = append(args, opts.Before)
	}

	return conditions, args
}

// GetStats returns index statistics.
func (s *Store) GetStats(ctx context.Context) (*IndexStats, error) {
	stats := &IndexStats{
		SessionsByAgent: make(map[AgentType]int),
	}

	// Get total sessions
	_ = s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM conv_sessions").Scan(&stats.TotalSessions)

	// Get total turns
	_ = s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM conv_turns").Scan(&stats.TotalTurns)

	// Get sessions by agent
	rows, err := s.db.QueryContext(ctx, "SELECT agent, COUNT(*) FROM conv_sessions GROUP BY agent")
	if err == nil {
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var agent string
			var count int
			if rows.Scan(&agent, &count) == nil {
				stats.SessionsByAgent[NormalizeAgentType(AgentType(agent))] += count
			}
		}
	}

	// Get last indexed time - scan as string first since SQLite stores as TEXT
	var lastIndexedStr sql.NullString
	if err := s.db.QueryRowContext(ctx, "SELECT MAX(created_at) FROM conv_sessions").Scan(&lastIndexedStr); err == nil && lastIndexedStr.Valid {
		// Try common SQLite timestamp formats
		for _, layout := range []string{
			"2006-01-02 15:04:05",
			"2006-01-02T15:04:05Z",
			"2006-01-02T15:04:05.000Z",
			time.RFC3339,
		} {
			if t, err := time.Parse(layout, lastIndexedStr.String); err == nil {
				stats.LastIndexed = t
				break
			}
		}
	}

	// Get database size
	if info, err := os.Stat(s.dbPath); err == nil {
		stats.IndexSizeBytes = info.Size()
	}

	return stats, nil
}

// GetAllTurnIDs returns all turn IDs that need embeddings.
func (s *Store) GetAllTurnIDs(ctx context.Context) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT t.id FROM conv_turns t
		LEFT JOIN conv_turn_embeddings e ON t.id = e.turn_id
		WHERE e.turn_id IS NULL
	`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}

	return ids, rows.Err()
}

// GetTurnContent retrieves turn content for embedding.
func (s *Store) GetTurnContent(ctx context.Context, turnID string) (string, error) {
	var content string
	err := s.db.QueryRowContext(ctx, `
		SELECT combined_content FROM conv_turns WHERE id = ?
	`, turnID).Scan(&content)
	return content, err
}

// GetTurnContentBatch retrieves content for multiple turns.
func (s *Store) GetTurnContentBatch(ctx context.Context, turnIDs []string) (map[string]string, error) {
	if len(turnIDs) == 0 {
		return nil, nil
	}

	placeholders := make([]string, len(turnIDs))
	args := make([]interface{}, len(turnIDs))
	for i, id := range turnIDs {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT id, combined_content FROM conv_turns WHERE id IN (%s)
	`, strings.Join(placeholders, ","))

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	result := make(map[string]string)
	for rows.Next() {
		var id, content string
		if err := rows.Scan(&id, &content); err != nil {
			return nil, err
		}
		result[id] = content
	}

	return result, rows.Err()
}

// MissingEmbeddingsCountForSession returns how many turns in a session lack embeddings.
func (s *Store) MissingEmbeddingsCountForSession(ctx context.Context, sessionID string) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM conv_turns t
		WHERE t.session_id = ?
		  AND NOT EXISTS (
			SELECT 1 FROM conv_turn_embeddings e
			WHERE e.turn_id = t.id
			   OR e.turn_id LIKE t.id || ':%'
		  )
	`, sessionID).Scan(&count)
	return count, err
}

// SessionExists checks if a session already exists.
func (s *Store) SessionExists(ctx context.Context, sessionID string) (bool, error) {
	var count int
	err := s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM conv_sessions WHERE id = ?", sessionID).Scan(&count)
	return count > 0, err
}

// SessionMeta contains lightweight session metadata for update checks.
type SessionMeta struct {
	TotalTurns int
	EndedAt    time.Time
}

// GetSessionMeta returns lightweight session metadata.
func (s *Store) GetSessionMeta(ctx context.Context, sessionID string) (SessionMeta, bool, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT total_turns, ended_at
		FROM conv_sessions WHERE id = ?
	`, sessionID)

	var meta SessionMeta
	var endedAt sql.NullTime
	if err := row.Scan(&meta.TotalTurns, &endedAt); err != nil {
		if err == sql.ErrNoRows {
			return SessionMeta{}, false, nil
		}
		return SessionMeta{}, false, err
	}
	if endedAt.Valid {
		meta.EndedAt = endedAt.Time
	}
	return meta, true, nil
}

// float32ToBlob converts a float32 slice to bytes.
func float32ToBlob(embedding []float32) []byte {
	buf := make([]byte, len(embedding)*4)
	for i, v := range embedding {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	return buf
}

// blobToFloat32 converts bytes to a float32 slice.
func blobToFloat32(blob []byte) []float32 {
	if len(blob)%4 != 0 {
		return nil
	}
	result := make([]float32, len(blob)/4)
	for i := range result {
		bits := uint32(blob[i*4]) |
			uint32(blob[i*4+1])<<8 |
			uint32(blob[i*4+2])<<16 |
			uint32(blob[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result
}

// cosineSimilarity computes the cosine similarity between two vectors.
// Returns a value between 0-1 (1 = identical, 0 = orthogonal).
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	similarity := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	// Clamp to [0, 1] to handle floating point errors
	if similarity < 0 {
		return 0.0
	}
	if similarity > 1 {
		return 1.0
	}
	return similarity
}
