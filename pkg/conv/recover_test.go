package conv

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

type fakeRecallRetriever struct {
	results        []SearchResult
	lexicalResults []SearchResult
	semanticErr    error
}

func (f *fakeRecallRetriever) RetrieveTurns(_ context.Context, _ string, opts SearchOptions) ([]SearchResult, error) {
	if opts.ExactMatch {
		return f.lexicalResults, nil
	}
	if f.semanticErr != nil {
		return nil, f.semanticErr
	}
	return f.results, nil
}

type sequenceRecallRetriever struct {
	results [][]SearchResult
	calls   int
}

func (s *sequenceRecallRetriever) RetrieveTurns(context.Context, string, SearchOptions) ([]SearchResult, error) {
	index := s.calls
	if index >= len(s.results) {
		index = len(s.results) - 1
	}
	s.calls++
	return s.results[index], nil
}

type fakeRecallSnapshot struct {
	generations []int64
	calls       int
	refreshed   bool
}

func (f *fakeRecallSnapshot) ConversationGeneration(context.Context) (int64, error) {
	index := f.calls
	if index >= len(f.generations) {
		index = len(f.generations) - 1
	}
	f.calls++
	return f.generations[index], nil
}

func (f *fakeRecallSnapshot) RefreshVectorSnapshot(context.Context) error {
	f.refreshed = true
	return nil
}

type fakeRecallReader struct {
	sessions map[string]*Session
	stats    IndexStats
}

func (f *fakeRecallReader) GetSession(_ context.Context, id string) (*Session, error) {
	session := f.sessions[id]
	if session == nil {
		return nil, errors.New("session not found")
	}
	return session, nil
}

func (f *fakeRecallReader) GetStats(context.Context) (*IndexStats, error) {
	stats := f.stats
	return &stats, nil
}

func TestRecallRetainsMatchesNeighborsAndTailAcrossAgents(t *testing.T) {
	now := time.Date(2026, 7, 23, 12, 0, 0, 0, time.UTC)
	reader := &fakeRecallReader{
		stats: IndexStats{TotalSessions: 2, LastIndexed: now},
		sessions: map[string]*Session{
			"codex-session": {
				ID: "codex-session", Agent: AgentCodexCLI, ProjectPath: "/repo/sgrep", ProjectName: "sgrep", StartedAt: now,
				Turns: []Turn{
					{Index: 0, UserContent: "initial request", AssistContent: "initial answer"},
					{Index: 1, UserContent: "incremental indexing question", AssistContent: "first decision"},
					{Index: 2, UserContent: "follow up", AssistContent: "supporting detail"},
					{Index: 3, UserContent: "remaining work", AssistContent: "open item"},
					{Index: 4, UserContent: "tests", AssistContent: "tests passed"},
				},
			},
			"claude-session": {
				ID: "claude-session", Agent: AgentClaudeCode, ProjectPath: "/repo/other", ProjectName: "other", StartedAt: now.Add(-time.Hour),
				Turns: []Turn{{Index: 0, UserContent: "indexing", AssistContent: "alternative approach"}},
			},
		},
	}
	retriever := &fakeRecallRetriever{results: []SearchResult{
		{TurnID: "codex-session:1", SessionID: "codex-session", TurnIndex: 1, UserContent: "incremental indexing question", AssistContent: "first decision"},
		{TurnID: "claude-session:0", SessionID: "claude-session", TurnIndex: 0, UserContent: "indexing", AssistContent: "alternative approach"},
		{TurnID: "codex-session:3", SessionID: "codex-session", TurnIndex: 3, UserContent: "remaining work", AssistContent: "open item"},
	}}
	service := &RecallService{turns: retriever, sessions: reader}

	response, err := service.Recall(context.Background(), "what remains from incremental indexing?", RecallOptions{MaxBytes: 24 * 1024, CurrentDir: "/repo/sgrep/pkg/conv"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Status != RecallOK {
		t.Fatalf("status = %q, want ok; warnings=%+v", response.Status, response.Warnings)
	}
	if len(response.Sessions) != 2 {
		t.Fatalf("sessions = %d, want 2", len(response.Sessions))
	}
	if response.Sessions[0].SessionID != "codex-session" || response.Sessions[0].Affinity != "current" {
		t.Fatalf("first session = %+v, want current codex session", response.Sessions[0])
	}
	if len(response.Sessions[0].Evidence) != 5 {
		t.Fatalf("codex evidence = %+v, want all five deduplicated turns", response.Sessions[0].Evidence)
	}
	var matchedTail bool
	for _, evidence := range response.Sessions[0].Evidence {
		if evidence.Citation == "" || !strings.HasPrefix(evidence.SourceRef, "conv://codex/") || !evidence.Untrusted {
			t.Fatalf("invalid evidence provenance: %+v", evidence)
		}
		if evidence.TurnIndex == 3 && containsString(evidence.Reasons, "match") && containsString(evidence.Reasons, "tail") {
			matchedTail = true
		}
	}
	if !matchedTail {
		t.Fatal("turn 3 did not retain both match and tail reasons")
	}
}

func TestRecallFallsBackToLexicalAndHonorsBudget(t *testing.T) {
	huge := strings.Repeat("λ historical output ", 2000)
	session := &Session{
		ID: "pi-session", Agent: AgentPiMono, ProjectPath: "/repo", StartedAt: time.Now(),
		Turns: []Turn{{Index: 0, UserContent: huge, AssistContent: huge}},
	}
	service := &RecallService{
		turns: &fakeRecallRetriever{
			semanticErr:    errors.New("embedding server unavailable"),
			lexicalResults: []SearchResult{{TurnID: "pi-session:0", SessionID: "pi-session", TurnIndex: 0, UserContent: huge, AssistContent: huge}},
		},
		sessions: &fakeRecallReader{
			stats:    IndexStats{TotalSessions: 1},
			sessions: map[string]*Session{"pi-session": session},
		},
	}

	response, err := service.Recall(context.Background(), "historical output", RecallOptions{MaxBytes: MinRecallMaxBytes, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Status != RecallPartial || response.RetrievalMode != "lexical_fallback" {
		t.Fatalf("unexpected fallback response: status=%s mode=%s", response.Status, response.RetrievalMode)
	}
	data, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(data) + 1; got > MinRecallMaxBytes {
		t.Fatalf("serialized response is %d bytes, budget is %d", got, MinRecallMaxBytes)
	}
	if len(response.Sessions) != 1 || len(response.Sessions[0].Evidence) != 1 || !response.Sessions[0].Evidence[0].Truncated {
		t.Fatalf("expected one truncated cited match, got %+v", response.Sessions)
	}
}

func TestRecallEmptyLexicalFallbackIsPartial(t *testing.T) {
	service := &RecallService{
		turns: &fakeRecallRetriever{semanticErr: errors.New("embedding server unavailable")},
		sessions: &fakeRecallReader{
			stats:    IndexStats{TotalSessions: 1},
			sessions: map[string]*Session{},
		},
	}

	response, err := service.Recall(context.Background(), "missing historical topic", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Status != RecallPartial || response.RetrievalMode != "lexical_fallback" || len(response.Sessions) != 0 {
		t.Fatalf("incomplete empty retrieval was reported as definitive: %+v", response)
	}
}

func TestRecallRetriesWhenMatchedTurnWasReplaced(t *testing.T) {
	session := &Session{ID: "session", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "new request", AssistContent: "new answer"}}}
	retriever := &sequenceRecallRetriever{results: [][]SearchResult{
		{{TurnID: "session:0", SessionID: "session", TurnIndex: 0, UserContent: "old request", AssistContent: "old answer"}},
		{{TurnID: "session:0", SessionID: "session", TurnIndex: 0, UserContent: "new request", AssistContent: "new answer"}},
	}}
	service := &RecallService{
		turns:    retriever,
		sessions: &fakeRecallReader{stats: IndexStats{TotalSessions: 1}, sessions: map[string]*Session{"session": session}},
	}
	response, err := service.Recall(context.Background(), "request", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	if retriever.calls != 2 {
		t.Fatalf("retrieval calls = %d, want one retry", retriever.calls)
	}
	if response.Status != RecallOK || len(response.Sessions) != 1 || response.Sessions[0].Evidence[0].User != "new request" {
		t.Fatalf("recall did not use one consistent snapshot: %+v", response)
	}
}

func TestRecallOmitsFirstAttemptWhenConsistencyRetryFails(t *testing.T) {
	session := &Session{ID: "session", Agent: AgentCodexCLI, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "new request", AssistContent: "new answer"}}}
	stale := SearchResult{TurnID: "session:0", SessionID: "session", TurnIndex: 0, UserContent: "old request", AssistContent: "old answer"}
	retriever := &sequenceRecallRetriever{results: [][]SearchResult{{stale}, {stale}}}
	service := &RecallService{
		turns:    retriever,
		sessions: &fakeRecallReader{stats: IndexStats{TotalSessions: 1}, sessions: map[string]*Session{"session": session}},
	}
	response, err := service.Recall(context.Background(), "request", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Status != RecallPartial || len(response.Sessions) != 0 {
		t.Fatalf("stale first-attempt evidence leaked after retry failure: %+v", response)
	}
}

func TestRecallRetriesEmptyResultsWhenIndexGenerationChanged(t *testing.T) {
	session := &Session{ID: "session", Agent: AgentPiMono, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "new topic", AssistContent: "new answer"}}}
	retriever := &sequenceRecallRetriever{results: [][]SearchResult{
		{},
		{{TurnID: "session:0", SessionID: "session", TurnIndex: 0, UserContent: "new topic", AssistContent: "new answer"}},
	}}
	snapshot := &fakeRecallSnapshot{generations: []int64{1, 2, 2, 2}}
	service := &RecallService{
		turns:    retriever,
		sessions: &fakeRecallReader{stats: IndexStats{TotalSessions: 1}, sessions: map[string]*Session{"session": session}},
		snapshot: snapshot,
	}
	response, err := service.Recall(context.Background(), "new topic", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	if response.Status != RecallOK || len(response.Sessions) != 1 || !snapshot.refreshed {
		t.Fatalf("generation retry failed: response=%+v refreshed=%v", response, snapshot.refreshed)
	}
}

func TestRecallEarlyResponsesHonorValidBudget(t *testing.T) {
	service := &RecallService{
		turns:    &fakeRecallRetriever{},
		sessions: &fakeRecallReader{stats: IndexStats{}},
	}
	response, err := service.Recall(context.Background(), strings.Repeat("\x00", 5000), RecallOptions{MaxBytes: MinRecallMaxBytes, CurrentDir: strings.Repeat("/long", 1000)})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(data) + 1; got > MinRecallMaxBytes {
		t.Fatalf("early response is %d bytes, budget is %d", got, MinRecallMaxBytes)
	}
	if !response.Budget.Truncated {
		t.Fatal("oversized early response did not report truncation")
	}
}

func TestRecallOversizedSessionMetadataStillHonorsBudget(t *testing.T) {
	id := strings.Repeat("session", 1000)
	project := "/" + strings.Repeat("project/", 1000)
	session := &Session{ID: id, Agent: AgentClaudeCode, ProjectPath: project, StartedAt: time.Now(), Turns: []Turn{{Index: 0, UserContent: "query", AssistContent: "answer"}}}
	service := &RecallService{
		turns:    &fakeRecallRetriever{results: []SearchResult{{TurnID: id + ":0", SessionID: id, TurnIndex: 0, UserContent: "query", AssistContent: "answer"}}},
		sessions: &fakeRecallReader{stats: IndexStats{TotalSessions: 1}, sessions: map[string]*Session{id: session}},
	}
	response, err := service.Recall(context.Background(), "query", RecallOptions{MaxBytes: MinRecallMaxBytes, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(response)
	if len(data)+1 > MinRecallMaxBytes {
		t.Fatalf("metadata-heavy response is %d bytes, budget is %d", len(data)+1, MinRecallMaxBytes)
	}
	if response.Status != RecallPartial || response.OmittedEvidence == 0 {
		t.Fatalf("metadata omission was not disclosed: %+v", response)
	}
}

func TestRecallIsDeterministic(t *testing.T) {
	now := time.Date(2026, 7, 23, 12, 0, 0, 0, time.UTC)
	session := &Session{ID: "session", Agent: AgentOpenCode, StartedAt: now, Turns: []Turn{{Index: 0, UserContent: "query", AssistContent: "answer"}}}
	service := &RecallService{
		turns:    &fakeRecallRetriever{results: []SearchResult{{TurnID: "session:0", SessionID: "session", TurnIndex: 0, UserContent: "query", AssistContent: "answer"}}},
		sessions: &fakeRecallReader{stats: IndexStats{TotalSessions: 1, LastIndexed: now}, sessions: map[string]*Session{"session": session}},
	}
	first, err := service.Recall(context.Background(), "query", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	second, err := service.Recall(context.Background(), "query", RecallOptions{MaxBytes: 8192, CurrentDir: "/repo"})
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := json.Marshal(first)
	secondJSON, _ := json.Marshal(second)
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("recall is not deterministic:\n%s\n%s", firstJSON, secondJSON)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
