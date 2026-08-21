package conv

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

type stubDocumentEmbedder struct {
	fail  bool
	calls int
}

func (s *stubDocumentEmbedder) EmbedDocuments(_ context.Context, documents []string) ([][]float32, error) {
	s.calls++
	if s.fail {
		return nil, errors.New("embedding failed")
	}
	embeddings := make([][]float32, len(documents))
	for i := range embeddings {
		embeddings[i] = make([]float32, defaultDims)
		embeddings[i][i%defaultDims] = 1
	}
	return embeddings, nil
}

func TestIndexerPublishesContentAddressedSessionSnapshotsAtomically(t *testing.T) {
	store, err := NewStore(StoreConfig{DBPath: filepath.Join(t.TempDir(), "conv.db"), Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	now := time.Now().UTC()
	original := &Session{
		ID: "snapshot", Agent: AgentCodexCLI, SourcePath: "/tmp/session.jsonl", StartedAt: now, EndedAt: now,
		Turns: []Turn{
			{Index: 0, UserContent: "old request", AssistContent: "old answer"},
			{Index: 1, UserContent: "removed request", AssistContent: "removed answer"},
		},
	}
	embedder := &stubDocumentEmbedder{}
	indexer := NewIndexer(IndexerConfig{Store: store, Embedder: embedder})
	result, err := indexer.IndexSessions(ctx, []*Session{original})
	if err != nil || len(result.Errors) != 0 || result.SessionsIndexed != 1 {
		t.Fatalf("initial index failed: result=%+v err=%v", result, err)
	}

	updated := &Session{
		ID: "snapshot", Agent: AgentCodexCLI, SourcePath: "/tmp/session.jsonl", StartedAt: now, EndedAt: now,
		Turns: []Turn{{Index: 0, UserContent: "new request", AssistContent: "new answer"}},
	}
	failing := NewIndexer(IndexerConfig{Store: store, Embedder: &stubDocumentEmbedder{fail: true}})
	failedResult, err := failing.IndexSessions(ctx, []*Session{updated})
	if err != nil {
		t.Fatal(err)
	}
	if len(failedResult.Errors) != 1 {
		t.Fatalf("embedding failure was not reported: %+v", failedResult)
	}
	stored, err := store.GetSession(ctx, "snapshot")
	if err != nil {
		t.Fatal(err)
	}
	if len(stored.Turns) != 2 || stored.Turns[0].UserContent != "old request" {
		t.Fatalf("failed refresh changed published snapshot: %+v", stored.Turns)
	}

	result, err = indexer.IndexSessions(ctx, []*Session{updated})
	if err != nil || len(result.Errors) != 0 || result.SessionsIndexed != 1 {
		t.Fatalf("updated index failed: result=%+v err=%v", result, err)
	}
	stored, err = store.GetSession(ctx, "snapshot")
	if err != nil {
		t.Fatal(err)
	}
	if len(stored.Turns) != 1 || stored.Turns[0].UserContent != "new request" {
		t.Fatalf("snapshot was not replaced: %+v", stored.Turns)
	}
	ids, _, err := store.ExportTurnEmbeddings(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "snapshot:0" {
		t.Fatalf("stale embeddings survived replacement: %v", ids)
	}
	meta, ok, err := store.GetSessionMeta(ctx, "snapshot")
	if err != nil || !ok || meta.ContentHash == "" {
		t.Fatalf("content hash was not persisted: meta=%+v ok=%v err=%v", meta, ok, err)
	}

	calls := embedder.calls
	result, err = indexer.IndexSessions(ctx, []*Session{updated})
	if err != nil || len(result.Errors) != 0 || result.SessionsIndexed != 0 {
		t.Fatalf("unchanged snapshot was not skipped: result=%+v err=%v", result, err)
	}
	if embedder.calls != calls {
		t.Fatalf("unchanged snapshot was embedded again: calls %d -> %d", calls, embedder.calls)
	}
}

func TestIndexerRejectsPublicationWithoutEmbedder(t *testing.T) {
	store, err := NewStore(StoreConfig{DBPath: filepath.Join(t.TempDir(), "conv.db"), Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	session := &Session{ID: "no-embedder", Agent: AgentClaudeCode, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "request", AssistContent: "answer"}}}
	indexer := NewIndexer(IndexerConfig{Store: store})
	result, err := indexer.IndexSessions(context.Background(), []*Session{session})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Errors) != 1 {
		t.Fatalf("nil embedder publication errors = %v, want one", result.Errors)
	}
	exists, err := store.SessionExists(context.Background(), session.ID)
	if err != nil {
		t.Fatal(err)
	}
	if exists {
		t.Fatal("nil embedder published a text-only conversation snapshot")
	}
}

func TestMissingEmbeddingsRequiresCanonicalTurnID(t *testing.T) {
	store, err := NewStore(StoreConfig{DBPath: filepath.Join(t.TempDir(), "conv.db"), Dims: defaultDims})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	session := &Session{ID: "legacy", Agent: AgentClaudeCode, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "request", AssistContent: "answer"}}}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatal(err)
	}
	if err := store.StoreTurnEmbedding(ctx, "legacy:0:0", make([]float32, defaultDims)); err != nil {
		t.Fatal(err)
	}
	missing, err := store.MissingEmbeddingsCountForSession(ctx, "legacy")
	if err != nil {
		t.Fatal(err)
	}
	if missing != 1 {
		t.Fatalf("legacy suffixed embedding satisfied canonical coverage: missing=%d", missing)
	}
}
