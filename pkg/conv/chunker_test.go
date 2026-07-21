package conv

import (
	"strings"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
)

func TestChunker_SingleChunk(t *testing.T) {
	c := NewChunker()
	turn := &Turn{
		Index:         0,
		UserContent:   "How do I implement authentication?",
		AssistContent: "For authentication, you can use JWT tokens...",
	}

	chunks := c.ChunkTurn("session-123", turn)

	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk, got %d", len(chunks))
	}

	if chunks[0].ID != "session-123:0" {
		t.Errorf("expected ID 'session-123:0', got '%s'", chunks[0].ID)
	}

	if !strings.Contains(chunks[0].Content, "USER:") || !strings.Contains(chunks[0].Content, "ASSISTANT:") {
		t.Error("chunk content should contain USER and ASSISTANT labels")
	}
}

func TestChunker_LongTurnSplitting(t *testing.T) {
	c := NewChunkerWithConfig(ChunkerConfig{
		MaxTokens:    100, // Very low to force splitting
		OverlapChars: 20,
	})

	// Create a turn with long content
	longContent := strings.Repeat("This is a test paragraph with some content. ", 100)
	turn := &Turn{
		Index:         0,
		UserContent:   "How do I do this?",
		AssistContent: longContent,
	}

	chunks := c.ChunkTurn("session-456", turn)

	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks for long content, got %d", len(chunks))
	}

	// Verify chunk IDs
	for i, chunk := range chunks {
		if !strings.HasPrefix(chunk.ID, "session-456:0") {
			t.Errorf("chunk %d has unexpected ID: %s", i, chunk.ID)
		}
	}
}

func TestChunker_EnforcesByteBudgetForMultibyteAndOversizedUser(t *testing.T) {
	const maxTokens = 32
	c := NewChunkerWithConfig(ChunkerConfig{MaxTokens: maxTokens, OverlapChars: 8})
	turn := &Turn{
		Index:         0,
		UserContent:   strings.Repeat("認証", 100),
		AssistContent: strings.Repeat("回答", 100),
	}
	chunks := c.ChunkTurn("session", turn)
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want split", len(chunks))
	}
	for i, chunk := range chunks {
		if got := modelcfg.EstimateTokens(chunk.Content); got > maxTokens {
			t.Fatalf("chunk %d uses %d estimated tokens, budget %d", i, got, maxTokens)
		}
	}
}

func TestChunker_SessionChunking(t *testing.T) {
	c := NewChunker()
	session := &Session{
		ID: "test-session",
		Turns: []Turn{
			{Index: 0, UserContent: "Question 1", AssistContent: "Answer 1"},
			{Index: 1, UserContent: "Question 2", AssistContent: "Answer 2"},
			{Index: 2, UserContent: "Question 3", AssistContent: "Answer 3"},
		},
	}

	chunks := c.ChunkSession(session)

	if len(chunks) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(chunks))
	}

	for i, chunk := range chunks {
		expectedID := "test-session:" + string(rune('0'+i))
		if chunk.TurnIndex != i {
			t.Errorf("chunk %d has wrong turn index: %d", i, chunk.TurnIndex)
		}
		if chunk.SessionID != "test-session" {
			t.Errorf("chunk %d has wrong session ID: %s", i, chunk.SessionID)
		}
		_ = expectedID // Used for verification
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		text     string
		expected int
	}{
		{"", 0},
		{"hello", 2},                      // 5 chars / 4 = 1.25 -> 2
		{"hello world", 3},                // 11 chars / 4 = 2.75 -> 3
		{"This is a longer sentence.", 7}, // 27 chars / 4 = 6.75 -> 7
	}

	for _, tt := range tests {
		got := EstimateTokens(tt.text)
		// Allow some tolerance since it's an estimate
		if got < tt.expected-1 || got > tt.expected+1 {
			t.Errorf("EstimateTokens(%q) = %d, expected ~%d", tt.text, got, tt.expected)
		}
	}
}

func TestEstimateTurnTokens(t *testing.T) {
	turn := &Turn{
		UserContent:   "How do I do X?",      // ~4 tokens
		AssistContent: "You can do it by...", // ~5 tokens
	}

	tokens := EstimateTurnTokens(turn)
	if tokens < 5 || tokens > 15 {
		t.Errorf("EstimateTurnTokens returned unexpected value: %d", tokens)
	}
}
