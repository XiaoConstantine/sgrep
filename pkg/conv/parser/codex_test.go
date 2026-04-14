package parser

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
)

func TestCodexParser_AgentType(t *testing.T) {
	p := NewCodexParser()
	if p.AgentType() != conv.AgentCodexCLI {
		t.Errorf("expected agent type %s, got %s", conv.AgentCodexCLI, p.AgentType())
	}
}

func TestCodexParser_ParseEmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")

	if err := os.WriteFile(testFile, []byte(""), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("expected 0 sessions, got %d", len(sessions))
	}
}

func TestCodexParser_ParseWithSessionMeta(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")

	now := time.Now().Format(time.RFC3339)

	// Create entries with proper Codex format
	sessionMeta := codexSessionMeta{
		ID:            "session-123",
		CWD:           "/path/to/project",
		ModelProvider: "openai",
	}
	sessionMetaJSON, _ := json.Marshal(sessionMeta)

	userMsg := codexMessage{Role: "user", Content: "How do I use Go?"}
	userMsgJSON, _ := json.Marshal(userMsg)

	assistMsg := codexMessage{Role: "assistant", Content: "Go is a statically typed language..."}
	assistMsgJSON, _ := json.Marshal(assistMsg)

	entries := []codexEntry{
		{
			Timestamp: now,
			Type:      "session_meta",
			Payload:   sessionMetaJSON,
		},
		{
			Timestamp: now,
			Type:      "message",
			Payload:   userMsgJSON,
		},
		{
			Timestamp: now,
			Type:      "message",
			Payload:   assistMsgJSON,
		},
	}

	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	enc := json.NewEncoder(f)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
	_ = f.Close()

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if len(sessions[0].Turns) != 1 {
		t.Errorf("expected 1 turn, got %d", len(sessions[0].Turns))
	}
	if sessions[0].ID != "session-123" {
		t.Errorf("expected session ID 'session-123', got '%s'", sessions[0].ID)
	}
}

func TestCodexParser_ParseResponseItemMessages(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")

	now := time.Now().Format(time.RFC3339)

	sessionMeta := codexSessionMeta{
		ID:            "session-response-item",
		CWD:           "/path/to/project",
		ModelProvider: "openai",
	}
	sessionMetaJSON, _ := json.Marshal(sessionMeta)

	userMsg := codexResponseMessage{
		Type: "message",
		Role: "user",
		Content: []codexMessageContent{
			{Type: "input_text", Text: "Hello from user"},
		},
	}
	userMsgJSON, _ := json.Marshal(userMsg)

	assistMsg := codexResponseMessage{
		Type: "message",
		Role: "assistant",
		Content: []codexMessageContent{
			{Type: "output_text", Text: "Hello from assistant"},
		},
	}
	assistMsgJSON, _ := json.Marshal(assistMsg)

	entries := []codexEntry{
		{Timestamp: now, Type: "session_meta", Payload: sessionMetaJSON},
		{Timestamp: now, Type: "response_item", Payload: userMsgJSON},
		{Timestamp: now, Type: "response_item", Payload: assistMsgJSON},
	}

	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	enc := json.NewEncoder(f)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
	_ = f.Close()

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if len(sessions[0].Turns) != 1 {
		t.Fatalf("expected 1 turn, got %d", len(sessions[0].Turns))
	}
	if sessions[0].Turns[0].UserContent != "Hello from user" {
		t.Errorf("unexpected user content: %q", sessions[0].Turns[0].UserContent)
	}
	if sessions[0].Turns[0].AssistContent != "Hello from assistant" {
		t.Errorf("unexpected assistant content: %q", sessions[0].Turns[0].AssistContent)
	}
}

func TestCodexParser_ParseMultipleTurns(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")

	now := time.Now().Format(time.RFC3339)

	var entries []codexEntry

	// Add messages for multiple turns
	messages := []codexMessage{
		{Role: "user", Content: "Question 1"},
		{Role: "assistant", Content: "Answer 1"},
		{Role: "user", Content: "Question 2"},
		{Role: "assistant", Content: "Answer 2"},
	}

	for _, msg := range messages {
		msgJSON, _ := json.Marshal(msg)
		entries = append(entries, codexEntry{
			Timestamp: now,
			Type:      "message",
			Payload:   msgJSON,
		})
	}

	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	enc := json.NewEncoder(f)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
	_ = f.Close()

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if len(sessions[0].Turns) != 2 {
		t.Errorf("expected 2 turns, got %d", len(sessions[0].Turns))
	}
}

func TestCodexParser_ParseCodeDetection(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.jsonl")

	now := time.Now().Format(time.RFC3339)

	userMsg := codexMessage{Role: "user", Content: "Show me Go code"}
	userMsgJSON, _ := json.Marshal(userMsg)

	assistMsg := codexMessage{
		Role:    "assistant",
		Content: "Here's some code:\n```go\nfunc main() {\n    fmt.Println(\"Hello\")\n}\n```",
	}
	assistMsgJSON, _ := json.Marshal(assistMsg)

	entries := []codexEntry{
		{Timestamp: now, Type: "message", Payload: userMsgJSON},
		{Timestamp: now, Type: "message", Payload: assistMsgJSON},
	}

	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	enc := json.NewEncoder(f)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
	_ = f.Close()

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !sessions[0].Turns[0].HasCode {
		t.Error("expected HasCode to be true")
	}
	if len(sessions[0].Turns[0].CodeLangs) != 1 || sessions[0].Turns[0].CodeLangs[0] != "go" {
		t.Errorf("expected CodeLangs to contain 'go', got %v", sessions[0].Turns[0].CodeLangs)
	}
}

func TestCodexParser_Discover(t *testing.T) {
	tmpDir := t.TempDir()

	// Create directory structure similar to Codex CLI
	sessionsDir := filepath.Join(tmpDir, "sessions", "2024", "01", "15")
	if err := os.MkdirAll(sessionsDir, 0755); err != nil {
		t.Fatalf("failed to create sessions dir: %v", err)
	}

	// Create a JSONL file
	jsonlFile := filepath.Join(sessionsDir, "session.jsonl")
	if err := os.WriteFile(jsonlFile, []byte("{}"), 0644); err != nil {
		t.Fatalf("failed to create jsonl file: %v", err)
	}

	p := NewCodexParserWithPath(tmpDir)
	paths, err := p.Discover()

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(paths) != 1 {
		t.Errorf("expected 1 path, got %d", len(paths))
	}
}

func TestCodexParser_ParseLargeJSONLLine(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "large.jsonl")

	now := time.Now().Format(time.RFC3339)
	largeContent := strings.Repeat("x", 2*1024*1024)

	userMsg := codexMessage{Role: "user", Content: largeContent}
	userMsgJSON, _ := json.Marshal(userMsg)
	assistMsg := codexMessage{Role: "assistant", Content: "ok"}
	assistMsgJSON, _ := json.Marshal(assistMsg)

	entries := []codexEntry{
		{Timestamp: now, Type: "message", Payload: userMsgJSON},
		{Timestamp: now, Type: "message", Payload: assistMsgJSON},
	}

	f, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	enc := json.NewEncoder(f)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
	_ = f.Close()

	p := NewCodexParserWithPath(tmpDir)
	sessions, err := p.Parse(testFile)
	if err != nil {
		t.Fatalf("unexpected error parsing large JSONL line: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if len(sessions[0].Turns) != 1 {
		t.Fatalf("expected 1 turn, got %d", len(sessions[0].Turns))
	}
	if sessions[0].Turns[0].UserContent != largeContent {
		t.Fatalf("expected large user content to round-trip, got length %d", len(sessions[0].Turns[0].UserContent))
	}
}
