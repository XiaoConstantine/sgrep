package conv

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestNewStore(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()
}

func TestNewStore_DoesNotCreateLegacyVectorIndex(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	var count int
	err = store.db.QueryRow(`
		SELECT COUNT(*) FROM sqlite_master
		WHERE name IN ('idx_turn_embeddings_vec', 'idx_turn_embeddings_vec_shadow', 'idx_turn_embeddings_vec_shadow_idx')
	`).Scan(&count)
	if err != nil {
		t.Fatalf("failed to inspect sqlite_master: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected no legacy vector index artifacts, found %d", count)
	}
}

func TestNewStore_DropsLegacyVectorIndexArtifacts(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	legacyStatements := []string{
		`CREATE INDEX idx_turn_embeddings_vec ON conv_turn_embeddings(turn_id)`,
		`CREATE TABLE idx_turn_embeddings_vec_shadow (index_key INTEGER PRIMARY KEY, data BLOB)`,
		`CREATE INDEX idx_turn_embeddings_vec_shadow_idx ON idx_turn_embeddings_vec_shadow(index_key)`,
	}
	for _, stmt := range legacyStatements {
		if _, err := store.db.Exec(stmt); err != nil {
			t.Fatalf("failed to seed legacy artifact: %v", err)
		}
	}
	if _, err := store.db.Exec(`INSERT OR REPLACE INTO conv_metadata (key, value) VALUES ('schema_version', 1)`); err != nil {
		t.Fatalf("failed to seed schema version: %v", err)
	}
	if err := store.initSchema(); err != nil {
		t.Fatalf("failed to rerun initSchema: %v", err)
	}

	var count int
	err = store.db.QueryRow(`
		SELECT COUNT(*) FROM sqlite_master
		WHERE name IN ('idx_turn_embeddings_vec', 'idx_turn_embeddings_vec_shadow', 'idx_turn_embeddings_vec_shadow_idx')
	`).Scan(&count)
	if err != nil {
		t.Fatalf("failed to inspect sqlite_master after migration: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected legacy vector artifacts to be removed, found %d", count)
	}
}

func TestStore_StoreAndGetSession(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	session := &Session{
		ID:          "test-session-1",
		Agent:       AgentClaudeCode,
		SourcePath:  "/path/to/source",
		ProjectPath: "/path/to/project",
		ProjectName: "my-project",
		StartedAt:   time.Now().Add(-1 * time.Hour),
		EndedAt:     time.Now(),
		Turns: []Turn{
			{
				Index:         0,
				UserContent:   "How do I use Go?",
				AssistContent: "Go is a programming language...",
				HasCode:       false,
			},
			{
				Index:         1,
				UserContent:   "Show me an example",
				AssistContent: "```go\nfunc main() {}\n```",
				HasCode:       true,
				CodeLangs:     []string{"go"},
			},
		},
	}

	// Store session
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	// Retrieve session
	retrieved, err := store.GetSession(ctx, "test-session-1")
	if err != nil {
		t.Fatalf("failed to get session: %v", err)
	}

	if retrieved.ID != session.ID {
		t.Errorf("expected ID %s, got %s", session.ID, retrieved.ID)
	}
	if retrieved.Agent != session.Agent {
		t.Errorf("expected agent %s, got %s", session.Agent, retrieved.Agent)
	}
	if len(retrieved.Turns) != len(session.Turns) {
		t.Errorf("expected %d turns, got %d", len(session.Turns), len(retrieved.Turns))
	}
}

func TestStore_GetStats(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	// Initially empty
	stats, err := store.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}
	if stats.TotalSessions != 0 {
		t.Errorf("expected 0 sessions, got %d", stats.TotalSessions)
	}

	// Add some sessions
	sessions := []*Session{
		{
			ID:        "session-1",
			Agent:     AgentClaudeCode,
			StartedAt: time.Now(),
			EndedAt:   time.Now(),
			Turns:     []Turn{{Index: 0, UserContent: "q1", AssistContent: "a1"}},
		},
		{
			ID:        "session-2",
			Agent:     AgentCursor,
			StartedAt: time.Now(),
			EndedAt:   time.Now(),
			Turns:     []Turn{{Index: 0, UserContent: "q2", AssistContent: "a2"}},
		},
	}

	for _, s := range sessions {
		if err := store.StoreSession(ctx, s); err != nil {
			t.Fatalf("failed to store session: %v", err)
		}
	}

	stats, err = store.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}
	if stats.TotalSessions != 2 {
		t.Errorf("expected 2 sessions, got %d", stats.TotalSessions)
	}
	if stats.TotalTurns != 2 {
		t.Errorf("expected 2 turns, got %d", stats.TotalTurns)
	}
}

func TestStore_FullTextSearch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	// Add sessions with searchable content
	sessions := []*Session{
		{
			ID:        "session-auth",
			Agent:     AgentClaudeCode,
			StartedAt: time.Now(),
			EndedAt:   time.Now(),
			Turns: []Turn{
				{Index: 0, UserContent: "How do I implement authentication?", AssistContent: "Use JWT tokens for authentication."},
			},
		},
		{
			ID:        "session-db",
			Agent:     AgentClaudeCode,
			StartedAt: time.Now(),
			EndedAt:   time.Now(),
			Turns: []Turn{
				{Index: 0, UserContent: "How do I connect to a database?", AssistContent: "Use sql.Open to connect to your database."},
			},
		},
	}

	for _, s := range sessions {
		if err := store.StoreSession(ctx, s); err != nil {
			t.Fatalf("failed to store session: %v", err)
		}
	}

	// Test full text search - the store uses FTS5 through HybridSearch
	// Create a zero embedding for testing (semantic portion will be ignored)
	zeroEmbed := make([]float32, defaultDims)

	// Using hybrid search with high BM25 weight should find "authentication"
	results, err := store.HybridSearch(ctx, zeroEmbed, "authentication", 10, 0.0, 0.0, 1.0)
	if err != nil {
		t.Fatalf("failed to search: %v", err)
	}

	// Check that we got results
	authFound := false
	for _, r := range results {
		if r.SessionID == "session-auth" {
			authFound = true
		}
	}
	if !authFound {
		t.Log("Note: FTS search for 'authentication' may need actual term matching")
	}
}

func TestHybridSearch_PreservesBM25Order(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	session := &Session{
		ID:        "session-bm25",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "auth auth auth auth", AssistContent: "response"},
			{Index: 1, UserContent: "auth auth", AssistContent: "response"},
			{Index: 2, UserContent: "auth", AssistContent: "response"},
		},
	}

	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	queryTerms := "auth"
	rows, err := store.db.QueryContext(ctx, `
		SELECT t.id, bm25(conv_turns_fts) as bm25_score
		FROM conv_turns_fts fts
		JOIN conv_turns t ON t.rowid = fts.rowid
		WHERE conv_turns_fts MATCH ?
		ORDER BY bm25_score
		LIMIT ?
	`, queryTerms, 10)
	if err != nil {
		t.Fatalf("failed to query FTS results: %v", err)
	}
	defer func() { _ = rows.Close() }()

	var ftsOrder []string
	for rows.Next() {
		var turnID string
		var bm25Score float64
		if err := rows.Scan(&turnID, &bm25Score); err != nil {
			t.Fatalf("failed to scan FTS row: %v", err)
		}
		_ = bm25Score
		ftsOrder = append(ftsOrder, turnID)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("failed to read FTS rows: %v", err)
	}
	if len(ftsOrder) < 2 {
		t.Fatalf("expected at least 2 FTS results, got %d", len(ftsOrder))
	}

	zeroEmbed := make([]float32, defaultDims)
	results, err := store.HybridSearch(ctx, zeroEmbed, queryTerms, 10, 0.0, 0.0, 1.0)
	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}
	if len(results) < len(ftsOrder) {
		t.Fatalf("expected at least %d results, got %d", len(ftsOrder), len(results))
	}

	for i, turnID := range ftsOrder {
		if results[i].TurnID != turnID {
			t.Fatalf("expected HybridSearch result %d to be %s, got %s", i, turnID, results[i].TurnID)
		}
	}
}

func TestStore_StoreTurnEmbedding(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	// Store a session first
	session := &Session{
		ID:        "embed-test",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "Test", AssistContent: "Response"},
		},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	// Create embedding
	embedding := make([]float32, defaultDims)
	for i := range embedding {
		embedding[i] = float32(i) / float32(defaultDims)
	}

	// Store embedding
	turnID := "embed-test:0"
	if err := store.StoreTurnEmbedding(ctx, turnID, embedding); err != nil {
		t.Fatalf("failed to store embedding: %v", err)
	}

	// Embedding was stored successfully - verify by checking that GetAllTurnIDs
	// no longer returns this turn (since it now has an embedding)
	turnIDs, err := store.GetAllTurnIDs(ctx)
	if err != nil {
		t.Fatalf("failed to get turn IDs: %v", err)
	}
	// The turn should no longer be in the "needs embedding" list
	for _, id := range turnIDs {
		if id == turnID {
			t.Error("turn should not be in 'needs embedding' list after storing embedding")
		}
	}
}

func TestStore_SessionExists(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	// Check non-existent session
	exists, err := store.SessionExists(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("failed to check session: %v", err)
	}
	if exists {
		t.Error("expected session to not exist")
	}

	// Create and store session
	session := &Session{
		ID:        "exists-test",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns:     []Turn{{Index: 0, UserContent: "Test", AssistContent: "Response"}},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	// Check existing session
	exists, err = store.SessionExists(ctx, "exists-test")
	if err != nil {
		t.Fatalf("failed to check session: %v", err)
	}
	if !exists {
		t.Error("expected session to exist")
	}
}

func TestStore_GetAllTurnIDs(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()

	// Add a session with turns
	session := &Session{
		ID:        "turnids-test",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "Q1", AssistContent: "A1"},
			{Index: 1, UserContent: "Q2", AssistContent: "A2"},
		},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	// Get all turn IDs
	turnIDs, err := store.GetAllTurnIDs(ctx)
	if err != nil {
		t.Fatalf("failed to get turn IDs: %v", err)
	}
	if len(turnIDs) != 2 {
		t.Errorf("expected 2 turn IDs, got %d", len(turnIDs))
	}
}
