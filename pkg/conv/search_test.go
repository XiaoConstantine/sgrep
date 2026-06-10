package conv

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestTruncate(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"short", 10, "short"},
		{"exactly10!", 10, "exactly10!"},
		{"this is a longer string", 10, "this is a..."}, // truncate breaks at word boundary
		{"", 10, ""},
	}

	for _, tt := range tests {
		result := truncate(tt.input, tt.maxLen)
		if result != tt.expected {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.maxLen, result, tt.expected)
		}
	}
}

func TestRelativeTime(t *testing.T) {
	now := time.Now()

	tests := []struct {
		input    time.Time
		expected string
	}{
		{now.Add(-30 * time.Second), "just now"},
		{now.Add(-5 * time.Minute), "5 minutes ago"},
		{now.Add(-2 * time.Hour), "2 hours ago"},
		{now.Add(-24 * time.Hour), "1 day ago"},
		{now.Add(-7 * 24 * time.Hour), "1 week ago"},
	}

	for _, tt := range tests {
		result := relativeTime(tt.input)
		if result != tt.expected {
			t.Errorf("relativeTime(%v) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestParseDuration(t *testing.T) {
	tests := []struct {
		input       string
		expectError bool
		checkValid  bool // Only check if result is non-zero
	}{
		{"1h", false, true},
		{"24h", false, true},
		{"7d", false, true},
		{"30d", false, true},
		{"1w", false, true},
		{"2w", false, true},
		{"1m", false, true},
		{"3m", false, true},
		{"1y", false, true},
		{"invalid", true, false},
	}

	for _, tt := range tests {
		result, err := ParseDuration(tt.input)
		if tt.expectError {
			if err == nil {
				t.Errorf("ParseDuration(%q) expected error, got nil", tt.input)
			}
		} else {
			if err != nil {
				t.Errorf("ParseDuration(%q) unexpected error: %v", tt.input, err)
			}
			if tt.checkValid && result == 0 {
				t.Errorf("ParseDuration(%q) returned zero duration", tt.input)
			}
		}
	}
}

func TestSearchOptions_Defaults(t *testing.T) {
	opts := DefaultSearchOptions()

	if opts.Limit != 10 {
		t.Errorf("expected default limit 10, got %d", opts.Limit)
	}
	if opts.Threshold != 0.5 {
		t.Errorf("expected default threshold 0.5, got %f", opts.Threshold)
	}
	if opts.Agent != AgentAll {
		t.Errorf("expected default agent AgentAll, got %s", opts.Agent)
	}
}

func TestDeduplicateBySession(t *testing.T) {
	results := []SearchResult{
		{
			SessionID: "session-1",
			Score:     0.8,
			TurnIndex: 0,
		},
		{
			SessionID: "session-1",
			Score:     0.9, // Higher score is better (similarity)
			TurnIndex: 1,
		},
		{
			SessionID: "session-2",
			Score:     0.85,
			TurnIndex: 0,
		},
	}

	deduped := deduplicateBySession(results)

	if len(deduped) != 2 {
		t.Errorf("expected 2 results after deduplication, got %d", len(deduped))
	}

	// First result should be session-1 with better score (0.9, higher is better)
	var session1Found, session2Found bool
	for _, r := range deduped {
		if r.SessionID == "session-1" {
			session1Found = true
			if r.Score != 0.9 {
				t.Errorf("expected session-1 to have score 0.9 (better), got %f", r.Score)
			}
		}
		if r.SessionID == "session-2" {
			session2Found = true
		}
	}
	if !session1Found {
		t.Error("session-1 not found in deduplicated results")
	}
	if !session2Found {
		t.Error("session-2 not found in deduplicated results")
	}
}

func TestSearcherExactUsesKeywordSearchWithoutEmbedder(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := NewStore(StoreConfig{DBPath: filepath.Join(tmpDir, "test.db"), Dims: defaultDims})
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	session := &Session{
		ID:          "session-auth",
		Agent:       AgentClaudeCode,
		ProjectPath: "/repo/auth",
		ProjectName: "auth",
		StartedAt:   time.Now(),
		EndedAt:     time.Now(),
		Turns: []Turn{
			{Index: 0, UserContent: "How do I implement authentication?", AssistContent: "Use JWT tokens for authentication."},
		},
	}
	if err := store.StoreSession(ctx, session); err != nil {
		t.Fatalf("failed to store session: %v", err)
	}

	searcher := NewSearcher(store, nil)
	opts := DefaultSearchOptions()
	opts.ExactMatch = true
	opts.Limit = 5

	response, err := searcher.Search(ctx, "authentication", opts)
	if err != nil {
		t.Fatalf("exact search failed: %v", err)
	}
	if len(response.Results) != 1 {
		t.Fatalf("expected 1 exact result, got %d", len(response.Results))
	}
	if response.Results[0].SessionID != "session-auth" {
		t.Fatalf("expected session-auth, got %s", response.Results[0].SessionID)
	}
	if response.Results[0].MatchType != "keyword" {
		t.Fatalf("expected keyword match type, got %q", response.Results[0].MatchType)
	}
}

func TestConversationHit_Fields(t *testing.T) {
	hit := ConversationHit{
		SessionID:   "test-session",
		Agent:       AgentClaudeCode,
		ProjectName: "my-project",
		Score:       0.95,
		MatchedTurn: TurnPreview{
			TurnIndex:  0,
			UserSnip:   "How do I do X?",
			AssistSnip: "You can do X by...",
		},
		Actions: HitActions{
			Resume:  "claude --resume test-session",
			View:    "sgrep conv view test-session",
			Export:  "sgrep conv export test-session -o conversation.md",
			Context: "sgrep conv context test-session",
		},
	}

	if hit.SessionID != "test-session" {
		t.Errorf("unexpected session ID: %s", hit.SessionID)
	}
	if hit.Agent != AgentClaudeCode {
		t.Errorf("unexpected agent: %s", hit.Agent)
	}
	if hit.Score != 0.95 {
		t.Errorf("unexpected score: %f", hit.Score)
	}
	if hit.Actions.Resume == "" {
		t.Error("expected non-empty resume command")
	}
	if hit.MatchedTurn.UserSnip != "How do I do X?" {
		t.Errorf("unexpected user snip: %s", hit.MatchedTurn.UserSnip)
	}
}

func TestSearchResponse_Fields(t *testing.T) {
	response := SearchResponse{
		Query:      "authentication",
		SearchTime: 100,
		TotalHits:  5,
		Returned:   5,
		Results:    []ConversationHit{},
	}

	if response.Query != "authentication" {
		t.Errorf("unexpected query: %s", response.Query)
	}
	if response.SearchTime != 100 {
		t.Errorf("unexpected search time: %d", response.SearchTime)
	}
}

func TestParseAgentType(t *testing.T) {
	tests := []struct {
		input    string
		expected AgentType
	}{
		{"claude", AgentClaudeCode},
		{"codex", AgentCodexCLI},
		{"cursor", AgentCursor},
		{"opencode", AgentOpenCode},
		{"pi", AgentPiMono},
		{"pi-mono", AgentPiMono},
		{"unknown", AgentAll},
		{"", AgentAll},
	}

	for _, tt := range tests {
		result := ParseAgentType(tt.input)
		if result != tt.expected {
			t.Errorf("ParseAgentType(%q) = %s, want %s", tt.input, result, tt.expected)
		}
	}
}
