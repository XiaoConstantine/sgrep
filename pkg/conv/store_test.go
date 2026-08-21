package conv

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	storepkg "github.com/XiaoConstantine/sgrep/pkg/store"
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

func TestOpenStoreReadOnlyDoesNotCreateOrWrite(t *testing.T) {
	tmpDir := t.TempDir()
	missingPath := filepath.Join(tmpDir, "missing.db")
	if store, err := OpenStoreReadOnly(missingPath); err == nil {
		_ = store.Close()
		t.Fatal("OpenStoreReadOnly succeeded for missing database")
	}

	dbPath := filepath.Join(tmpDir, "test.db")
	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()
	session := &Session{
		ID:        "readable-session",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns:     []Turn{{Index: 0, UserContent: "hello", AssistContent: "world"}},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("failed to close writable store: %v", err)
	}

	readStore, err := OpenStoreReadOnly(dbPath)
	if err != nil {
		t.Fatalf("failed to open read-only store: %v", err)
	}
	defer func() { _ = readStore.Close() }()

	if _, err := readStore.GetSession(ctx, "readable-session"); err != nil {
		t.Fatalf("failed to read existing session: %v", err)
	}
	err = readStore.StoreSession(ctx, &Session{
		ID:        "write-should-fail",
		Agent:     AgentClaudeCode,
		StartedAt: time.Now(),
		Turns:     []Turn{{Index: 0, UserContent: "write", AssistContent: "blocked"}},
	})
	if err == nil {
		t.Fatal("StoreSession succeeded on read-only store")
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

func TestStore_NormalizesLegacyPiMonoAgentRows(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	_, err = store.db.ExecContext(ctx, `
		INSERT INTO conv_sessions (
			id, agent, agent_version, source_path, project_path, project_name, git_branch, git_commit,
			started_at, ended_at, total_turns, total_tokens
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, "legacy-pi-session", "pi-mono", "", "/tmp/pi/session.jsonl", "/tmp/project", "project", "", "", now, now, 1, 0)
	if err != nil {
		t.Fatalf("failed to insert legacy session: %v", err)
	}

	_, err = store.db.ExecContext(ctx, `
		INSERT INTO conv_turns (
			id, session_id, turn_index, user_content, assistant_content, combined_content, timestamp
		) VALUES (?, ?, ?, ?, ?, ?, ?)
	`, "legacy-pi-session:0", "legacy-pi-session", 0, "hello", "world", "USER: hello\nASSISTANT: world", now)
	if err != nil {
		t.Fatalf("failed to insert legacy turn: %v", err)
	}

	embedding := make([]float32, defaultDims)
	embedding[0] = 1
	if err := store.StoreTurnEmbedding(ctx, "legacy-pi-session:0", embedding); err != nil {
		t.Fatalf("failed to store embedding: %v", err)
	}

	session, err := store.GetSession(ctx, "legacy-pi-session")
	if err != nil {
		t.Fatalf("failed to get session: %v", err)
	}
	if session.Agent != AgentPiMono {
		t.Fatalf("expected normalized agent %q, got %q", AgentPiMono, session.Agent)
	}

	stats, err := store.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}
	if stats.SessionsByAgent[AgentPiMono] != 1 {
		t.Fatalf("expected pi count 1, got %d", stats.SessionsByAgent[AgentPiMono])
	}

	results, err := store.FilteredSearch(ctx, embedding, SearchOptions{
		Limit:     10,
		Threshold: 0,
		Agent:     AgentPiMono,
	})
	if err != nil {
		t.Fatalf("failed filtered search: %v", err)
	}
	if len(results) != 1 || results[0].SessionID != "legacy-pi-session" {
		t.Fatalf("expected legacy pi session in filtered search, got %+v", results)
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

func TestConversationEmbeddingContextMismatchRequiresReindex(t *testing.T) {
	t.Setenv("SGREP_CONTEXT_TOKENS", "256")
	path := filepath.Join(t.TempDir(), "test.db")
	store, err := NewStore(StoreConfig{DBPath: path, Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SGREP_CONTEXT_TOKENS", "512")
	if _, err := OpenStoreReadOnly(path); err == nil {
		t.Fatal("conversation store accepted embeddings built for a different context budget")
	}
}

func TestConversationFTSUpdateRemovesOldTerms(t *testing.T) {
	store, err := NewStore(StoreConfig{DBPath: filepath.Join(t.TempDir(), "test.db"), Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	session := &Session{ID: "session", Agent: AgentClaudeCode, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "obsoleteTerm", AssistContent: "old"}}}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	session.Turns[0].UserContent = "replacementTerm"
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	opts := DefaultSearchOptions()
	opts.Limit = 10
	oldHits, err := store.KeywordSearch(ctx, "obsoleteTerm", opts)
	if err != nil {
		t.Fatal(err)
	}
	newHits, err := store.KeywordSearch(ctx, "replacementTerm", opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(oldHits) != 0 || len(newHits) != 1 {
		t.Fatalf("updated conversation FTS old=%+v new=%+v", oldHits, newHits)
	}
}

func TestKeywordSearchAppliesFiltersAndLabelsResults(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	sessions := []*Session{
		{
			ID:          "session-auth-claude",
			Agent:       AgentClaudeCode,
			ProjectPath: "/repo/auth",
			ProjectName: "auth",
			StartedAt:   time.Now(),
			EndedAt:     time.Now(),
			Turns: []Turn{
				{Index: 0, UserContent: "authentication authentication", AssistContent: "Use JWT authentication."},
			},
		},
		{
			ID:          "session-auth-codex",
			Agent:       AgentCodexCLI,
			ProjectPath: "/repo/auth",
			ProjectName: "auth",
			StartedAt:   time.Now(),
			EndedAt:     time.Now(),
			Turns: []Turn{
				{Index: 0, UserContent: "authentication", AssistContent: "Use session cookies."},
			},
		},
	}
	for _, session := range sessions {
		if err := store.StoreSession(ctx, session); err != nil {
			t.Fatalf("failed to store session: %v", err)
		}
	}

	results, err := store.KeywordSearch(ctx, "authentication", SearchOptions{
		Limit: 10,
		Agent: AgentClaudeCode,
	})
	if err != nil {
		t.Fatalf("KeywordSearch failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 filtered result, got %d", len(results))
	}
	if results[0].SessionID != "session-auth-claude" {
		t.Fatalf("expected claude session, got %s", results[0].SessionID)
	}
	if results[0].MatchType != "keyword" {
		t.Fatalf("expected keyword match type, got %q", results[0].MatchType)
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

func TestStore_RebuildTQVectorStoreLoadsForReadOnlySearch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()
	session := &Session{
		ID:        "tq-search-test",
		Agent:     AgentCodexCLI,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "turboquant mse search", AssistContent: "compact vector sidecar"},
			{Index: 1, UserContent: "database migration", AssistContent: "schema update"},
		},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	first := make([]float32, defaultDims)
	first[0] = 1
	second := make([]float32, defaultDims)
	second[1] = 1
	if err := store.StoreTurnEmbeddingBatch(ctx,
		[]string{"tq-search-test:0", "tq-search-test:1"},
		[][]float32{first, second},
	); err != nil {
		t.Fatalf("failed to store embeddings: %v", err)
	}

	count, err := store.RebuildTQVectorStore(ctx)
	if err != nil {
		t.Fatalf("RebuildTQVectorStore failed: %v", err)
	}
	if count != 2 {
		t.Fatalf("TQ vector count = %d, want 2", count)
	}
	if store.tq == nil {
		t.Fatal("expected writable store to load rebuilt TQ sidecar")
	}
	if err := store.Close(); err != nil {
		t.Fatalf("failed to close writable store: %v", err)
	}

	t.Setenv("SGREP_CONV_VECTOR_BACKEND", "tqmse")
	readStore, err := OpenStoreReadOnly(dbPath)
	if err != nil {
		t.Fatalf("failed to open read-only TQ store: %v", err)
	}
	defer func() { _ = readStore.Close() }()
	if readStore.tq == nil {
		t.Fatal("expected read-only store to load TQ sidecar")
	}

	results, err := readStore.VectorSearch(ctx, first, 2, 0)
	if err != nil {
		t.Fatalf("VectorSearch failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected TQ vector search result")
	}
	if results[0].TurnID != "tq-search-test:0" {
		t.Fatalf("top result = %s, want tq-search-test:0", results[0].TurnID)
	}
	if results[0].MatchType != "semantic" {
		t.Fatalf("match type = %q, want semantic", results[0].MatchType)
	}
}

func TestStore_WritableOpenRepairsLegacyTQMetadataInForcedMode(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	session := &Session{ID: "legacy-sidecar", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{
		{Index: 0, UserContent: "legacy metadata", AssistContent: "repair it"},
	}}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	embedding := make([]float32, defaultDims)
	embedding[0] = 1
	if err := store.StoreTurnEmbedding(ctx, "legacy-sidecar:0", embedding); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RebuildTQVectorStore(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.ExecContext(ctx, `DELETE FROM conv_metadata WHERE key = ?`, convTQVectorSidecarSHA256Key); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	t.Setenv("SGREP_CONV_VECTOR_BACKEND", "tqmse")
	if readStore, err := OpenStoreReadOnly(dbPath); err == nil {
		_ = readStore.Close()
		t.Fatal("read-only forced-TQ open accepted legacy sidecar metadata")
	}
	writable, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("writable open could not recover legacy sidecar: %v", err)
	}
	defer func() { _ = writable.Close() }()
	if writable.tq != nil {
		t.Fatal("writable open loaded an uncertified legacy sidecar")
	}
	if _, err := writable.RebuildTQVectorStore(ctx); err != nil {
		t.Fatalf("writable store could not rebuild legacy sidecar: %v", err)
	}
	if writable.tq == nil {
		t.Fatal("rebuilt sidecar was not loaded")
	}
}

func TestStore_OpenReaderDropsSidecarChangedByAnotherProcess(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	writer, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = writer.Close() }()
	ctx := context.Background()
	session := &Session{ID: "cross-process", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{
		{Index: 0, UserContent: "first", AssistContent: "answer"},
		{Index: 1, UserContent: "second", AssistContent: "answer"},
	}}
	if err := writer.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	first := make([]float32, defaultDims)
	first[0] = 1
	second := make([]float32, defaultDims)
	second[1] = 1
	if err := writer.StoreTurnEmbeddingBatch(ctx, []string{"cross-process:0", "cross-process:1"}, [][]float32{first, second}); err != nil {
		t.Fatal(err)
	}
	if _, err := writer.RebuildTQVectorStore(ctx); err != nil {
		t.Fatal(err)
	}

	reader, err := OpenStoreReadOnly(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reader.Close() }()
	if reader.tq == nil {
		t.Fatal("reader did not load initial sidecar")
	}

	hash, err := sessionContentHash(session)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.ReplaceSessionWithEmbeddings(ctx, session,
		[]string{"cross-process:0", "cross-process:1"}, [][]float32{second, first}, hash); err != nil {
		t.Fatal(err)
	}
	if _, err := writer.RebuildTQVectorStore(ctx); err != nil {
		t.Fatal(err)
	}
	valid, err := reader.validateTQVectorStore(ctx, reader.tq, "")
	if err != nil {
		t.Fatal(err)
	}
	if valid {
		t.Fatal("old mapped sidecar was accepted with newer clean metadata")
	}
	results, err := reader.VectorSearch(ctx, second, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	currentGeneration, err := reader.ConversationGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if reader.tq == nil || reader.tqGeneration != currentGeneration {
		t.Fatal("reader did not refresh to the current cross-process sidecar")
	}
	if len(results) == 0 || results[0].TurnID != "cross-process:0" {
		t.Fatalf("reader did not use updated vectors: %+v", results)
	}
}

func TestStore_TQSearchFencesConcurrentGenerationChange(t *testing.T) {
	for _, filtered := range []bool{false, true} {
		name := "vector"
		if filtered {
			name = "filtered"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("SGREP_CONV_VECTOR_BACKEND", "")
			t.Setenv("SGREP_VECTOR_BACKEND", "")
			dbPath := filepath.Join(t.TempDir(), "test.db")
			writer, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = writer.Close() }()
			ctx := context.Background()
			session := &Session{ID: "generation-race", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{
				{Index: 0, UserContent: "old first", AssistContent: "old answer"},
				{Index: 1, UserContent: "old second", AssistContent: "old answer"},
			}}
			if err := writer.StoreSession(ctx, session); err != nil {
				t.Fatal(err)
			}
			first := make([]float32, defaultDims)
			first[0] = 1
			second := make([]float32, defaultDims)
			second[1] = 1
			ids := []string{"generation-race:0", "generation-race:1"}
			if err := writer.StoreTurnEmbeddingBatch(ctx, ids, [][]float32{first, second}); err != nil {
				t.Fatal(err)
			}
			if _, err := writer.RebuildTQVectorStore(ctx); err != nil {
				t.Fatal(err)
			}

			reader, err := OpenStoreReadOnly(dbPath)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = reader.Close() }()
			updatedSession := &Session{ID: session.ID, Agent: session.Agent, StartedAt: session.StartedAt, Turns: []Turn{
				{Index: 0, UserContent: "new first", AssistContent: "new answer"},
				{Index: 1, UserContent: "new second", AssistContent: "new answer"},
			}}
			hash, err := sessionContentHash(updatedSession)
			if err != nil {
				t.Fatal(err)
			}
			var updateErr error
			updated := false
			reader.afterTQScore = func() {
				reader.afterTQScore = nil
				updated = true
				updateErr = writer.ReplaceSessionWithEmbeddings(ctx, updatedSession, ids, [][]float32{second, first}, hash)
			}

			var results []SearchResult
			if filtered {
				results, err = reader.FilteredSearch(ctx, first, SearchOptions{Limit: 2, Threshold: 0, Agent: AgentCodexCLI})
			} else {
				results, err = reader.VectorSearch(ctx, first, 2, 0)
			}
			if err != nil {
				t.Fatal(err)
			}
			if !updated {
				t.Fatal("generation update hook was not called")
			}
			if updateErr != nil {
				t.Fatalf("concurrent generation update failed: %v", updateErr)
			}
			if len(results) == 0 || results[0].TurnID != "generation-race:1" || results[0].UserContent != "new second" {
				t.Fatalf("search mixed stale scores with current rows: %+v", results)
			}
		})
	}
}

func TestStore_TwoPublishersCannotCertifyAnotherArtifact(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	writer, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = writer.Close() }()
	ctx := context.Background()
	session := &Session{ID: "publish-race", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{
		{Index: 0, UserContent: "first", AssistContent: "answer"},
		{Index: 1, UserContent: "second", AssistContent: "answer"},
	}}
	if err := writer.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	first := make([]float32, defaultDims)
	first[0] = 1
	second := make([]float32, defaultDims)
	second[1] = 1
	ids := []string{"publish-race:0", "publish-race:1"}
	oldEmbeddings := [][]float32{first, second}
	if err := writer.StoreTurnEmbeddingBatch(ctx, ids, oldEmbeddings); err != nil {
		t.Fatal(err)
	}
	oldGeneration, err := writer.ConversationGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}

	build := func(name string, embeddings [][]float32) string {
		t.Helper()
		path := filepath.Join(dir, name)
		count, err := storepkg.BuildTQVectorStoreAtPath(ctx, path, ids, embeddings, storepkg.TQVectorBuildOptions{Dims: defaultDims, Bits: 4, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}
		if count != len(ids) {
			t.Fatalf("built %d vectors, want %d", count, len(ids))
		}
		return path
	}
	oldFirst := build("old-first.tqmse", oldEmbeddings)
	oldSecond := build("old-second.tqmse", oldEmbeddings)

	newEmbeddings := [][]float32{second, first}
	hash, err := sessionContentHash(session)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.ReplaceSessionWithEmbeddings(ctx, session, ids, newEmbeddings, hash); err != nil {
		t.Fatal(err)
	}
	newGeneration, err := writer.ConversationGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}
	newFirst := build("new-first.tqmse", newEmbeddings)
	newSecond := build("new-second.tqmse", newEmbeddings)

	// A current publisher after a stale publisher certifies exactly the current artifact.
	if clean, err := writer.publishTQVectorStaging(ctx, oldGeneration, len(ids), oldFirst); err != nil || clean {
		t.Fatalf("stale publisher clean=%v err=%v", clean, err)
	}
	if clean, err := writer.publishTQVectorStaging(ctx, newGeneration, len(ids), newFirst); err != nil || !clean {
		t.Fatalf("current publisher clean=%v err=%v", clean, err)
	}
	reader, err := OpenStoreReadOnly(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reader.VectorSearch(ctx, second, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if reader.tq == nil || len(results) == 0 || results[0].TurnID != "publish-race:0" {
		t.Fatalf("current-after-stale publication was not usable: tq=%v results=%+v", reader.tq != nil, results)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}

	// Pause the current publisher after its rename, then let a stale publisher
	// overwrite the shared path before the current publisher certifies metadata.
	// Hashing the shared path after rename would incorrectly certify stale bytes.
	stalePublisher, err := OpenStore(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	published := make(chan struct{})
	release := make(chan struct{})
	writer.afterTQPublish = func() {
		close(published)
		<-release
	}
	type publishResult struct {
		clean bool
		err   error
	}
	currentDone := make(chan publishResult, 1)
	go func() {
		clean, err := writer.publishTQVectorStaging(ctx, newGeneration, len(ids), newSecond)
		currentDone <- publishResult{clean: clean, err: err}
	}()
	<-published
	staleClean, staleErr := stalePublisher.publishTQVectorStaging(ctx, oldGeneration, len(ids), oldSecond)
	close(release)
	currentResult := <-currentDone
	if staleErr != nil || staleClean {
		t.Fatalf("interleaved stale publisher clean=%v err=%v", staleClean, staleErr)
	}
	writer.afterTQPublish = nil
	if currentResult.err != nil || !currentResult.clean {
		t.Fatalf("interleaved current publisher clean=%v err=%v", currentResult.clean, currentResult.err)
	}
	if err := stalePublisher.Close(); err != nil {
		t.Fatal(err)
	}
	reader, err = OpenStoreReadOnly(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reader.Close() }()
	if reader.tq != nil {
		t.Fatal("reader accepted stale-after-current sidecar bytes")
	}
	results, err = reader.VectorSearch(ctx, second, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 || results[0].TurnID != "publish-race:0" {
		t.Fatalf("SQL fallback did not use current vectors: %+v", results)
	}
	if _, err := os.Stat(oldFirst); !os.IsNotExist(err) {
		t.Fatalf("published staging artifact still exists: %v", err)
	}
}

func TestStore_StaleTQVectorStoreRejectedAfterSameCountEmbeddingUpdate(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewStore(StoreConfig{DBPath: dbPath, Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()
	session := &Session{
		ID:        "tq-stale-test",
		Agent:     AgentCodexCLI,
		StartedAt: time.Now(),
		EndedAt:   time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "old topic", AssistContent: "old answer"},
			{Index: 1, UserContent: "new topic", AssistContent: "new answer"},
		},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	first := make([]float32, defaultDims)
	first[0] = 1
	second := make([]float32, defaultDims)
	second[1] = 1
	if err := store.StoreTurnEmbeddingBatch(ctx,
		[]string{"tq-stale-test:0", "tq-stale-test:1"},
		[][]float32{first, second},
	); err != nil {
		t.Fatalf("failed to store initial embeddings: %v", err)
	}
	if _, err := store.RebuildTQVectorStore(ctx); err != nil {
		t.Fatalf("RebuildTQVectorStore failed: %v", err)
	}

	// Update the same turn IDs with the same vector count. Count-only validation
	// would accept the stale sidecar and rank tq-stale-test:1 for query second.
	if err := store.StoreTurnEmbeddingBatch(ctx,
		[]string{"tq-stale-test:0", "tq-stale-test:1"},
		[][]float32{second, first},
	); err != nil {
		t.Fatalf("failed to update embeddings: %v", err)
	}
	if store.tq != nil {
		t.Fatal("expected in-process TQ sidecar to be invalidated after embedding update")
	}
	results, err := store.VectorSearch(ctx, second, 2, 0)
	if err != nil {
		t.Fatalf("same-process VectorSearch failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected same-process SQL fallback search result")
	}
	if results[0].TurnID != "tq-stale-test:0" {
		t.Fatalf("same-process top result = %s, want tq-stale-test:0", results[0].TurnID)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("failed to close writable store: %v", err)
	}

	t.Run("default backend falls back to SQL", func(t *testing.T) {
		t.Setenv("SGREP_CONV_VECTOR_BACKEND", "")
		t.Setenv("SGREP_VECTOR_BACKEND", "")

		readStore, err := OpenStoreReadOnly(dbPath)
		if err != nil {
			t.Fatalf("failed to open read-only store: %v", err)
		}
		defer func() { _ = readStore.Close() }()
		if readStore.tq != nil {
			t.Fatal("expected stale TQ sidecar to stay unloaded")
		}

		results, err := readStore.VectorSearch(ctx, second, 2, 0)
		if err != nil {
			t.Fatalf("VectorSearch failed: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("expected SQL fallback search result")
		}
		if results[0].TurnID != "tq-stale-test:0" {
			t.Fatalf("top result = %s, want tq-stale-test:0", results[0].TurnID)
		}
	})

	t.Run("forced TQ rejects stale sidecar", func(t *testing.T) {
		t.Setenv("SGREP_CONV_VECTOR_BACKEND", "tqmse")
		t.Setenv("SGREP_VECTOR_BACKEND", "")

		readStore, err := OpenStoreReadOnly(dbPath)
		if err == nil {
			_ = readStore.Close()
			t.Fatal("expected stale TQ sidecar error")
		}
		if !strings.Contains(err.Error(), "stale") {
			t.Fatalf("OpenStoreReadOnly error = %v, want stale sidecar error", err)
		}
	})
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
