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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
	storepkg "github.com/XiaoConstantine/sgrep/pkg/store"
)

const (
	// Schema version for migrations
	schemaVersion = 2
	// Default embedding dimensions for nomic-embed-text
	defaultDims = 768
	// Compact TQ-MSE sidecar for conversation turn embeddings.
	convTQVectorFileName = "turn_embeddings.tqmse"

	convTQVectorGenerationKey        = "tq_vector_generation"
	convTQVectorSidecarGenerationKey = "tq_vector_sidecar_generation"
	convTQVectorSidecarCountKey      = "tq_vector_sidecar_count"
	convEmbeddingFormatVersionKey    = "embedding_format_version"
	convEmbeddingContextTokensKey    = "embedding_context_tokens"
)

// Store handles conversation storage and retrieval.
type Store struct {
	db     *sql.DB
	dbPath string
	dims   int

	tqPath string
	tqMu   sync.RWMutex
	tq     *storepkg.TQVectorStore
}

type tqVectorMetadata struct {
	generation           int64
	sidecarGeneration    int64
	sidecarCount         int64
	hasSidecarGeneration bool
	hasSidecarCount      bool
}

type sqlQuerier interface {
	QueryContext(context.Context, string, ...interface{}) (*sql.Rows, error)
}

type sqlExecutor interface {
	ExecContext(context.Context, string, ...interface{}) (sql.Result, error)
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
	if err := store.loadTQIfAvailable(context.Background()); err != nil {
		_ = store.Close()
		return nil, err
	}

	return store, nil
}

// OpenStore opens an existing store.
func OpenStore(dbPath string) (*Store, error) {
	store, err := openStore(dbPath, defaultDims, false)
	if err != nil {
		return nil, err
	}
	if err := store.validateEmbeddingFormat(); err != nil {
		_ = store.Close()
		return nil, err
	}
	if err := store.loadTQIfAvailable(context.Background()); err != nil {
		_ = store.Close()
		return nil, err
	}
	return store, nil
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
	if err := store.validateEmbeddingFormat(); err != nil {
		_ = store.Close()
		return nil, err
	}
	if err := store.loadTQIfAvailable(context.Background()); err != nil {
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
		db:     db,
		dbPath: dbPath,
		dims:   dims,
		tqPath: filepath.Join(filepath.Dir(dbPath), convTQVectorFileName),
	}, nil
}

func (s *Store) validateEmbeddingFormat() error {
	var version int
	if err := s.db.QueryRow(`SELECT CAST(value AS INTEGER) FROM conv_metadata WHERE key = ?`, convEmbeddingFormatVersionKey).Scan(&version); err != nil {
		return fmt.Errorf("conversation embedding format is unknown; run 'sgrep conv index --force'")
	}
	if version != modelcfg.EmbeddingFormatVersion {
		return fmt.Errorf("conversation embedding format is version %d, need %d; run 'sgrep conv index --force'", version, modelcfg.EmbeddingFormatVersion)
	}
	var contextTokens int
	if err := s.db.QueryRow(`SELECT CAST(value AS INTEGER) FROM conv_metadata WHERE key = ?`, convEmbeddingContextTokensKey).Scan(&contextTokens); err != nil {
		return fmt.Errorf("conversation embedding context is unknown; run 'sgrep conv index --force'")
	}
	if contextTokens != modelcfg.ContextTokens() {
		return fmt.Errorf("conversation embedding context is %d tokens, need %d; run 'sgrep conv index --force'", contextTokens, modelcfg.ContextTokens())
	}
	return nil
}

func (s *Store) finalizeEmbeddingFormat(ctx context.Context) error {
	var missing int
	if err := s.db.QueryRowContext(ctx, `
		SELECT COUNT(*)
		FROM conv_turns t
		LEFT JOIN conv_turn_embeddings e ON e.turn_id = t.id
		WHERE e.turn_id IS NULL OR e.embedding IS NULL
	`).Scan(&missing); err != nil {
		return fmt.Errorf("verify conversation embedding coverage: %w", err)
	}
	if missing != 0 {
		return fmt.Errorf("conversation embedding rebuild incomplete: %d turns have no embedding", missing)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	for key, value := range map[string]int{
		convEmbeddingFormatVersionKey: modelcfg.EmbeddingFormatVersion,
		convEmbeddingContextTokensKey: modelcfg.ContextTokens(),
	} {
		if _, err := tx.ExecContext(ctx, `INSERT OR REPLACE INTO conv_metadata (key, value) VALUES (?, ?)`, key, value); err != nil {
			return fmt.Errorf("finalize conversation embedding metadata: %w", err)
		}
	}
	return tx.Commit()
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
	s.tqMu.Lock()
	tq := s.tq
	s.tq = nil
	s.tqMu.Unlock()

	var err error
	if tq != nil {
		err = tq.Close()
	}
	if closeErr := s.db.Close(); err == nil {
		err = closeErr
	}
	return err
}

// TQVectorPath returns the compact TQ-MSE sidecar path for turn embeddings.
func (s *Store) TQVectorPath() string {
	return s.tqPath
}

func conversationVectorBackend() string {
	if backend := strings.ToLower(os.Getenv("SGREP_CONV_VECTOR_BACKEND")); backend != "" {
		return backend
	}
	return strings.ToLower(os.Getenv("SGREP_VECTOR_BACKEND"))
}

func (s *Store) loadTQIfAvailable(ctx context.Context) error {
	backend := conversationVectorBackend()
	if backend == "sqlite" || backend == "libsql" {
		return nil
	}
	if s.tqPath == "" || !storepkg.HasTQVectorStoreAtPath(s.tqPath) {
		if backend == "tqmse" {
			return fmt.Errorf("SGREP_CONV_VECTOR_BACKEND=tqmse but %s is missing", s.tqPath)
		}
		return nil
	}

	tq, err := storepkg.OpenTQVectorStoreAtPath(s.tqPath)
	if err != nil {
		if backend == "tqmse" {
			return err
		}
		return nil
	}

	valid, err := s.validateTQVectorStore(ctx, tq, backend)
	if err != nil || !valid {
		_ = tq.Close()
		return err
	}

	s.tqMu.Lock()
	valid, err = s.validateTQVectorStore(ctx, tq, backend)
	if err != nil || !valid {
		s.tqMu.Unlock()
		_ = tq.Close()
		return err
	}
	old := s.tq
	s.tq = tq
	s.tqMu.Unlock()
	if old != nil {
		_ = old.Close()
	}
	return nil
}

func (s *Store) validateTQVectorStore(ctx context.Context, tq *storepkg.TQVectorStore, backend string) (bool, error) {
	count, err := s.turnEmbeddingCount(ctx)
	if err != nil {
		if backend == "tqmse" {
			return false, fmt.Errorf("check conversation embedding count: %w", err)
		}
		return false, nil
	}
	if count != tq.VectorCount() {
		if backend == "tqmse" {
			return false, fmt.Errorf("conversation TQ-MSE vector count %d does not match SQL embeddings %d", tq.VectorCount(), count)
		}
		return false, nil
	}
	metadata, err := s.tqVectorMetadata(ctx)
	if err != nil {
		if backend == "tqmse" {
			return false, fmt.Errorf("check conversation TQ-MSE metadata: %w", err)
		}
		return false, nil
	}
	if !metadata.sidecarClean(count) {
		if backend == "tqmse" {
			return false, fmt.Errorf("conversation TQ-MSE vector store is stale; run sgrep conv index")
		}
		return false, nil
	}
	return true, nil
}

func (s *Store) turnEmbeddingCount(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM conv_turn_embeddings WHERE embedding IS NOT NULL`).Scan(&count)
	return count, err
}

func (m tqVectorMetadata) sidecarClean(sqlCount int) bool {
	return m.hasSidecarGeneration &&
		m.hasSidecarCount &&
		m.generation == m.sidecarGeneration &&
		m.sidecarCount == int64(sqlCount)
}

func (s *Store) tqVectorMetadata(ctx context.Context) (tqVectorMetadata, error) {
	return tqVectorMetadataFrom(ctx, s.db)
}

func tqVectorMetadataFrom(ctx context.Context, q sqlQuerier) (tqVectorMetadata, error) {
	rows, err := q.QueryContext(ctx, `
		SELECT key, value
		FROM conv_metadata
		WHERE key IN (?, ?, ?)
	`, convTQVectorGenerationKey, convTQVectorSidecarGenerationKey, convTQVectorSidecarCountKey)
	if err != nil {
		return tqVectorMetadata{}, err
	}
	defer func() { _ = rows.Close() }()

	var metadata tqVectorMetadata
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return tqVectorMetadata{}, err
		}
		parsed, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return tqVectorMetadata{}, fmt.Errorf("parse %s=%q: %w", key, value, err)
		}
		switch key {
		case convTQVectorGenerationKey:
			metadata.generation = parsed
		case convTQVectorSidecarGenerationKey:
			metadata.sidecarGeneration = parsed
			metadata.hasSidecarGeneration = true
		case convTQVectorSidecarCountKey:
			metadata.sidecarCount = parsed
			metadata.hasSidecarCount = true
		}
	}
	if err := rows.Err(); err != nil {
		return tqVectorMetadata{}, err
	}
	return metadata, nil
}

func bumpTQVectorGeneration(ctx context.Context, exec sqlExecutor) error {
	if _, err := exec.ExecContext(ctx, `
		INSERT OR IGNORE INTO conv_metadata (key, value)
		VALUES (?, '0')
	`, convTQVectorGenerationKey); err != nil {
		return err
	}
	_, err := exec.ExecContext(ctx, `
		UPDATE conv_metadata
		SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
		WHERE key = ?
	`, convTQVectorGenerationKey)
	return err
}

func (s *Store) markTQVectorSidecarClean(ctx context.Context, generation int64, count int) (bool, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return false, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	if _, err := tx.ExecContext(ctx, `
		INSERT OR IGNORE INTO conv_metadata (key, value)
		VALUES (?, '0')
	`, convTQVectorGenerationKey); err != nil {
		return false, err
	}
	metadata, err := tqVectorMetadataFrom(ctx, tx)
	if err != nil {
		return false, err
	}
	if metadata.generation != generation {
		return false, nil
	}
	if err := setMetadataValue(ctx, tx, convTQVectorSidecarGenerationKey, strconv.FormatInt(generation, 10)); err != nil {
		return false, err
	}
	if err := setMetadataValue(ctx, tx, convTQVectorSidecarCountKey, strconv.Itoa(count)); err != nil {
		return false, err
	}
	if err := tx.Commit(); err != nil {
		return false, err
	}
	committed = true
	return true, nil
}

func setMetadataValue(ctx context.Context, exec sqlExecutor, key, value string) error {
	_, err := exec.ExecContext(ctx, `
		INSERT OR REPLACE INTO conv_metadata (key, value)
		VALUES (?, ?)
	`, key, value)
	return err
}

func (s *Store) closeTQLocked() error {
	if s.tq == nil {
		return nil
	}
	err := s.tq.Close()
	s.tq = nil
	return err
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

	var embeddingVersion, embeddingContext int
	_ = s.db.QueryRow(`SELECT CAST(value AS INTEGER) FROM conv_metadata WHERE key = ?`, convEmbeddingFormatVersionKey).Scan(&embeddingVersion)
	_ = s.db.QueryRow(`SELECT CAST(value AS INTEGER) FROM conv_metadata WHERE key = ?`, convEmbeddingContextTokensKey).Scan(&embeddingContext)
	if embeddingVersion != modelcfg.EmbeddingFormatVersion || embeddingContext != modelcfg.ContextTokens() {
		if _, err := s.db.Exec(`DELETE FROM conv_turn_embeddings`); err != nil {
			return fmt.Errorf("clear incompatible conversation embeddings: %w", err)
		}
		_ = os.Remove(s.tqPath)
		var turnCount int
		if err := s.db.QueryRow(`SELECT COUNT(*) FROM conv_turns`).Scan(&turnCount); err != nil {
			return fmt.Errorf("count conversation turns during embedding migration: %w", err)
		}
		version := 0 // Existing turns require a successful rebuild before search.
		if turnCount == 0 {
			version = modelcfg.EmbeddingFormatVersion
		}
		if _, err := s.db.Exec(`
			INSERT OR REPLACE INTO conv_metadata (key, value) VALUES (?, ?)
		`, convEmbeddingFormatVersionKey, version); err != nil {
			return fmt.Errorf("store conversation embedding migration state: %w", err)
		}
		if _, err := s.db.Exec(`
			INSERT OR REPLACE INTO conv_metadata (key, value) VALUES (?, ?)
		`, convEmbeddingContextTokensKey, modelcfg.ContextTokens()); err != nil {
			return fmt.Errorf("store conversation context migration state: %w", err)
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
			INSERT INTO conv_turns (
				id, session_id, turn_index, user_content, assistant_content, combined_content,
				timestamp, has_code, code_langs, parent_uuid, is_sidechain
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				session_id=excluded.session_id, turn_index=excluded.turn_index,
				user_content=excluded.user_content, assistant_content=excluded.assistant_content,
				combined_content=excluded.combined_content, timestamp=excluded.timestamp,
				has_code=excluded.has_code, code_langs=excluded.code_langs,
				parent_uuid=excluded.parent_uuid, is_sidechain=excluded.is_sidechain
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
	s.tqMu.Lock()
	defer s.tqMu.Unlock()

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

	if err := bumpTQVectorGeneration(ctx, tx); err != nil {
		return fmt.Errorf("mark conversation TQ-MSE vectors stale: %w", err)
	}

	blob := float32ToBlob(embedding)
	if sqliteDriverName == "libsql" {
		_, err = tx.ExecContext(ctx, `
			INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding)
			VALUES (?, vector32(?))
		`, turnID, blob)
	} else {
		// For sqlite3, store raw blob
		_, err = tx.ExecContext(ctx, `
			INSERT OR REPLACE INTO conv_turn_embeddings (turn_id, embedding)
			VALUES (?, ?)
		`, turnID, blob)
	}
	if err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	committed = true
	_ = s.closeTQLocked()
	return nil
}

// StoreTurnEmbeddingBatch stores embeddings for multiple turns.
func (s *Store) StoreTurnEmbeddingBatch(ctx context.Context, turnIDs []string, embeddings [][]float32) error {
	if len(turnIDs) != len(embeddings) {
		return fmt.Errorf("turnIDs and embeddings length mismatch")
	}
	if len(turnIDs) == 0 {
		return nil
	}

	s.tqMu.Lock()
	defer s.tqMu.Unlock()

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

	if err := bumpTQVectorGeneration(ctx, tx); err != nil {
		return fmt.Errorf("mark conversation TQ-MSE vectors stale: %w", err)
	}

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
		return err
	}
	committed = true
	_ = s.closeTQLocked()
	return nil
}

// ExportTurnEmbeddings returns all turn embeddings for compact TQ-MSE export.
func (s *Store) ExportTurnEmbeddings(ctx context.Context) ([]string, [][]float32, error) {
	return exportTurnEmbeddingsFrom(ctx, s.db, s.dims)
}

func (s *Store) exportTurnEmbeddingsSnapshot(ctx context.Context) (int64, []string, [][]float32, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, nil, nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()

	metadata, err := tqVectorMetadataFrom(ctx, tx)
	if err != nil {
		return 0, nil, nil, err
	}
	ids, embeddings, err := exportTurnEmbeddingsFrom(ctx, tx, s.dims)
	if err != nil {
		return 0, nil, nil, err
	}
	if err := tx.Commit(); err != nil {
		return 0, nil, nil, err
	}
	committed = true
	return metadata.generation, ids, embeddings, nil
}

func exportTurnEmbeddingsFrom(ctx context.Context, q sqlQuerier, dims int) ([]string, [][]float32, error) {
	rows, err := q.QueryContext(ctx, `
		SELECT turn_id, embedding
		FROM conv_turn_embeddings
		WHERE embedding IS NOT NULL
		ORDER BY turn_id
	`)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = rows.Close() }()

	ids := make([]string, 0)
	embeddings := make([][]float32, 0)
	for rows.Next() {
		var id string
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, nil, err
		}
		embedding := blobToFloat32(blob)
		if len(embedding) != dims {
			continue
		}
		ids = append(ids, id)
		embeddings = append(embeddings, embedding)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	return ids, embeddings, nil
}

// RebuildTQVectorStore refreshes the compact TQ-MSE turn-vector sidecar from SQL embeddings.
func (s *Store) RebuildTQVectorStore(ctx context.Context) (int, error) {
	generation, ids, embeddings, err := s.exportTurnEmbeddingsSnapshot(ctx)
	if err != nil {
		return 0, fmt.Errorf("export conversation turn embeddings: %w", err)
	}

	s.tqMu.Lock()
	if err := s.closeTQLocked(); err != nil {
		s.tqMu.Unlock()
		return 0, err
	}
	s.tqMu.Unlock()

	if len(ids) == 0 {
		if err := storepkg.RemoveTQVectorStoreAtPath(s.tqPath); err != nil {
			return 0, err
		}
		clean, err := s.markTQVectorSidecarClean(ctx, generation, 0)
		if err != nil {
			return 0, err
		}
		if !clean {
			return 0, fmt.Errorf("conversation embeddings changed while refreshing compact TQ-MSE vectors")
		}
		if err := s.finalizeEmbeddingFormat(ctx); err != nil {
			return 0, err
		}
		return 0, nil
	}

	count, err := storepkg.BuildTQVectorStoreAtPath(ctx, s.tqPath, ids, embeddings, storepkg.TQVectorBuildOptions{
		Dims: s.dims,
		Bits: 4,
		Seed: 42,
	})
	if err != nil {
		_ = s.loadTQIfAvailable(ctx)
		return 0, err
	}
	clean, err := s.markTQVectorSidecarClean(ctx, generation, count)
	if err != nil {
		return 0, err
	}
	if !clean {
		return count, fmt.Errorf("conversation embeddings changed while refreshing compact TQ-MSE vectors")
	}
	if err := s.loadTQIfAvailable(ctx); err != nil {
		return 0, err
	}
	if err := s.finalizeEmbeddingFormat(ctx); err != nil {
		return 0, err
	}
	return count, nil
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
	if results, ok, err := s.tqVectorSearch(ctx, embedding, limit, threshold); ok || err != nil {
		return results, err
	}

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

func (s *Store) tqVectorSearch(ctx context.Context, embedding []float32, limit int, threshold float64) ([]SearchResult, bool, error) {
	s.tqMu.RLock()
	defer s.tqMu.RUnlock()
	tq := s.tq
	if tq == nil {
		return nil, false, nil
	}

	distanceThreshold := 1.0 - threshold
	if threshold <= 0 {
		distanceThreshold = 2.0
	}
	hits, err := tq.Search(ctx, embedding, limit, distanceThreshold)
	if err != nil {
		return nil, true, err
	}
	if len(hits) == 0 {
		return nil, true, nil
	}

	ids := make([]string, len(hits))
	for i, hit := range hits {
		ids[i] = hit.ID
	}
	resultsByID, err := s.loadSearchResultsByTurnID(ctx, ids)
	if err != nil {
		return nil, true, err
	}

	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		result, ok := resultsByID[hit.ID]
		if !ok {
			continue
		}
		result.Score = clampSimilarity(1.0 - hit.Distance)
		result.MatchType = "semantic"
		results = append(results, result)
	}
	return results, true, nil
}

func (s *Store) loadSearchResultsByTurnID(ctx context.Context, ids []string) (map[string]SearchResult, error) {
	results := make(map[string]SearchResult, len(ids))
	const batchSize = 500
	for start := 0; start < len(ids); start += batchSize {
		end := start + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[start:end]
		placeholders := make([]string, len(batch))
		args := make([]interface{}, len(batch))
		for i, id := range batch {
			placeholders[i] = "?"
			args[i] = id
		}
		rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
			SELECT id, session_id, turn_index, user_content, assistant_content
			FROM conv_turns
			WHERE id IN (%s)
		`, strings.Join(placeholders, ",")), args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var result SearchResult
			if err := rows.Scan(&result.TurnID, &result.SessionID, &result.TurnIndex, &result.UserContent, &result.AssistContent); err != nil {
				_ = rows.Close()
				return nil, err
			}
			results[result.TurnID] = result
		}
		if err := rows.Err(); err != nil {
			_ = rows.Close()
			return nil, err
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
	}
	return results, nil
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
	if results, ok, err := s.tqFilteredSearch(ctx, embedding, opts, conditions, args); ok || err != nil {
		return results, err
	}

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

func (s *Store) tqFilteredSearch(ctx context.Context, embedding []float32, opts SearchOptions, conditions []string, args []interface{}) ([]SearchResult, bool, error) {
	s.tqMu.RLock()
	tq := s.tq
	if tq == nil {
		s.tqMu.RUnlock()
		return nil, false, nil
	}
	s.tqMu.RUnlock()

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}
	query := fmt.Sprintf(`
		SELECT t.id
		FROM conv_turns t
		JOIN conv_sessions sess ON t.session_id = sess.id
		%s
	`, whereClause)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, true, err
	}

	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			_ = rows.Close()
			return nil, true, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return nil, true, err
	}
	if err := rows.Close(); err != nil {
		return nil, true, err
	}
	if len(ids) == 0 {
		return nil, true, nil
	}

	s.tqMu.RLock()
	defer s.tqMu.RUnlock()
	tq = s.tq
	if tq == nil {
		return nil, false, nil
	}
	distances, err := tq.ScoreByID(ctx, embedding, ids)
	if err != nil {
		return nil, true, err
	}

	type scoredID struct {
		id    string
		score float64
	}
	scored := make([]scoredID, 0, len(distances))
	for id, distance := range distances {
		similarity := clampSimilarity(1.0 - distance)
		if similarity < opts.Threshold {
			continue
		}
		scored = append(scored, scoredID{id: id, score: similarity})
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})
	if len(scored) > opts.Limit {
		scored = scored[:opts.Limit]
	}
	if len(scored) == 0 {
		return nil, true, nil
	}

	topIDs := make([]string, len(scored))
	for i, item := range scored {
		topIDs[i] = item.id
	}
	resultsByID, err := s.loadSearchResultsByTurnID(ctx, topIDs)
	if err != nil {
		return nil, true, err
	}

	results := make([]SearchResult, 0, len(scored))
	for _, item := range scored {
		result, ok := resultsByID[item.id]
		if !ok {
			continue
		}
		result.Score = item.score
		result.MatchType = "semantic"
		results = append(results, result)
	}
	return results, true, nil
}

// FilteredHybridSearch combines filtered dense and FTS results.
func (s *Store) FilteredHybridSearch(ctx context.Context, embedding []float32, queryTerms string, opts SearchOptions) ([]SearchResult, error) {
	if strings.TrimSpace(queryTerms) == "" {
		return s.FilteredSearch(ctx, embedding, opts)
	}

	fetchOpts := opts
	fetchOpts.Limit = opts.Limit * 2
	if fetchOpts.Limit < 1 {
		fetchOpts.Limit = opts.Limit
	}

	vecResults, err := s.FilteredSearch(ctx, embedding, fetchOpts)
	if err != nil {
		return nil, err
	}
	ftsResults, err := s.KeywordSearch(ctx, queryTerms, fetchOpts)
	if err != nil {
		return vecResults, nil
	}

	combined := make(map[string]*SearchResult)
	rrfScores := make(map[string]float64)
	const k = 60.0

	for i, r := range vecResults {
		result := r
		combined[r.TurnID] = &result
		rrfScores[r.TurnID] = opts.SemanticWeight / (k + float64(i+1))
	}
	for i, r := range ftsResults {
		if _, ok := combined[r.TurnID]; !ok {
			result := r
			combined[r.TurnID] = &result
		}
		rrfScores[r.TurnID] += opts.BM25Weight / (k + float64(i+1))
	}

	results := make([]SearchResult, 0, len(combined))
	for turnID, result := range combined {
		result.Score = rrfScores[turnID]
		result.MatchType = "hybrid"
		results = append(results, *result)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > opts.Limit {
		results = results[:opts.Limit]
	}
	return results, nil
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
	if info, err := os.Stat(s.tqPath); err == nil {
		stats.IndexSizeBytes += info.Size()
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
	return clampSimilarity(similarity)
}

func clampSimilarity(similarity float64) float64 {
	if similarity < 0 {
		return 0.0
	}
	if similarity > 1 {
		return 1.0
	}
	return similarity
}
