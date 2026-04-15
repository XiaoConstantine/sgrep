package parser

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
)

func TestPiMonoParser_AgentType(t *testing.T) {
	p := NewPiMonoParser()
	if p.AgentType() != conv.AgentPiMono {
		t.Fatalf("expected agent type %s, got %s", conv.AgentPiMono, p.AgentType())
	}
}

func TestPiMonoParser_Discover(t *testing.T) {
	tmpDir := t.TempDir()
	basePath := filepath.Join(tmpDir, ".pi", "agent")
	sessionDir := filepath.Join(basePath, "sessions", "--tmp-project--")
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		t.Fatalf("failed to create session dir: %v", err)
	}

	sessionPath := filepath.Join(sessionDir, "2026-04-14T12-00-00Z_session.jsonl")
	if err := os.WriteFile(sessionPath, []byte(`{"type":"session","id":"session-1","timestamp":"2026-04-14T12:00:00Z","cwd":"/tmp/project"}`+"\n"), 0644); err != nil {
		t.Fatalf("failed to write session file: %v", err)
	}

	p := NewPiMonoParserWithPath(basePath)
	paths, err := p.Discover()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(paths) != 1 {
		t.Fatalf("expected 1 path, got %d", len(paths))
	}
	if paths[0] != sessionPath {
		t.Fatalf("unexpected path %q", paths[0])
	}
}

func TestPiMonoParser_ParseSingleSession(t *testing.T) {
	tmpDir := t.TempDir()
	basePath := filepath.Join(tmpDir, ".pi", "agent")
	sessionDir := filepath.Join(basePath, "sessions", "--tmp-project--")
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		t.Fatalf("failed to create session dir: %v", err)
	}

	sessionPath := filepath.Join(sessionDir, "session.jsonl")
	header := piSessionHeader{
		Type:      "session",
		Version:   3,
		ID:        "session-1",
		Timestamp: "2026-04-14T12:00:00Z",
		CWD:       "/tmp/project",
	}
	user := piSessionEntry{
		Type:      "message",
		ID:        "user-1",
		ParentID:  nil,
		Timestamp: "2026-04-14T12:00:01Z",
		Message: &piMessage{
			Role:    "user",
			Content: "How do I run the benchmarks?",
		},
	}
	assistant := piSessionEntry{
		Type:      "message",
		ID:        "assistant-1",
		ParentID:  strPtr("user-1"),
		Timestamp: "2026-04-14T12:00:02Z",
		Message: &piMessage{
			Role: "assistant",
			Content: []map[string]any{
				{"type": "thinking", "thinking": "ignored"},
				{"type": "text", "text": "Run `go test ./...` first."},
			},
		},
	}

	writePiJSONL(t, sessionPath, header, user, assistant)

	p := NewPiMonoParserWithPath(basePath)
	sessions, err := p.Parse(sessionPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if sessions[0].Agent != conv.AgentPiMono {
		t.Fatalf("expected agent %s, got %s", conv.AgentPiMono, sessions[0].Agent)
	}
	if sessions[0].ProjectPath != "/tmp/project" {
		t.Fatalf("unexpected project path %q", sessions[0].ProjectPath)
	}
	if len(sessions[0].Turns) != 1 {
		t.Fatalf("expected 1 turn, got %d", len(sessions[0].Turns))
	}
	if sessions[0].Turns[0].UserContent != "How do I run the benchmarks?" {
		t.Fatalf("unexpected user content %q", sessions[0].Turns[0].UserContent)
	}
	if sessions[0].Turns[0].AssistContent != "Run `go test ./...` first." {
		t.Fatalf("unexpected assistant content %q", sessions[0].Turns[0].AssistContent)
	}
}

func TestPiMonoParser_ParseUsesActiveLeafPath(t *testing.T) {
	tmpDir := t.TempDir()
	basePath := filepath.Join(tmpDir, ".pi", "agent")
	sessionDir := filepath.Join(basePath, "sessions", "--tmp-project--")
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		t.Fatalf("failed to create session dir: %v", err)
	}

	sessionPath := filepath.Join(sessionDir, "session.jsonl")
	header := piSessionHeader{
		Type:      "session",
		Version:   3,
		ID:        "session-branch",
		Timestamp: "2026-04-14T12:00:00Z",
		CWD:       "/tmp/project",
	}
	entries := []piSessionEntry{
		{
			Type:      "message",
			ID:        "u1",
			Timestamp: "2026-04-14T12:00:01Z",
			Message:   &piMessage{Role: "user", Content: "Question one"},
		},
		{
			Type:      "message",
			ID:        "a1",
			ParentID:  strPtr("u1"),
			Timestamp: "2026-04-14T12:00:02Z",
			Message:   &piMessage{Role: "assistant", Content: "Answer one"},
		},
		{
			Type:      "message",
			ID:        "u2-old",
			ParentID:  strPtr("a1"),
			Timestamp: "2026-04-14T12:00:03Z",
			Message:   &piMessage{Role: "user", Content: "Abandoned branch"},
		},
		{
			Type:      "message",
			ID:        "a2-old",
			ParentID:  strPtr("u2-old"),
			Timestamp: "2026-04-14T12:00:04Z",
			Message:   &piMessage{Role: "assistant", Content: "Old answer"},
		},
		{
			Type:      "message",
			ID:        "u2-new",
			ParentID:  strPtr("a1"),
			Timestamp: "2026-04-14T12:00:05Z",
			Message:   &piMessage{Role: "user", Content: "Active branch"},
		},
		{
			Type:      "message",
			ID:        "a2-new",
			ParentID:  strPtr("u2-new"),
			Timestamp: "2026-04-14T12:00:06Z",
			Message:   &piMessage{Role: "assistant", Content: "New answer"},
		},
	}

	writePiJSONL(t, sessionPath, append([]any{header}, entriesToAny(entries)...)...)

	p := NewPiMonoParserWithPath(basePath)
	sessions, err := p.Parse(sessionPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	if len(sessions[0].Turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(sessions[0].Turns))
	}
	if sessions[0].Turns[1].UserContent != "Active branch" {
		t.Fatalf("unexpected second turn user content %q", sessions[0].Turns[1].UserContent)
	}
	if sessions[0].Turns[1].AssistContent != "New answer" {
		t.Fatalf("unexpected second turn assistant content %q", sessions[0].Turns[1].AssistContent)
	}
}

func TestPiMonoParser_ParseSkipsMalformedLines(t *testing.T) {
	tmpDir := t.TempDir()
	basePath := filepath.Join(tmpDir, ".pi", "agent")
	sessionDir := filepath.Join(basePath, "sessions", "--tmp-project--")
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		t.Fatalf("failed to create session dir: %v", err)
	}

	sessionPath := filepath.Join(sessionDir, "session.jsonl")
	lines := []string{
		`{"type":"session","id":"session-1","timestamp":"2026-04-14T12:00:00Z","cwd":"/tmp/project"}`,
		`not json`,
		`{"type":"message","id":"u1","parentId":null,"timestamp":"2026-04-14T12:00:01Z","message":{"role":"user","content":"hello"}}`,
		`{"type":"message","id":"a1","parentId":"u1","timestamp":"2026-04-14T12:00:02Z","message":{"role":"assistant","content":"hi"}}`,
	}
	if err := os.WriteFile(sessionPath, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatalf("failed to write session file: %v", err)
	}

	p := NewPiMonoParserWithPath(basePath)
	sessions, err := p.Parse(sessionPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 1 || len(sessions[0].Turns) != 1 {
		t.Fatalf("expected 1 session with 1 turn, got %d sessions / %d turns", len(sessions), len(sessions[0].Turns))
	}
}

func writePiJSONL(t *testing.T, path string, entries ...any) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create session file: %v", err)
	}
	defer func() { _ = file.Close() }()

	enc := json.NewEncoder(file)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			t.Fatalf("failed to encode entry: %v", err)
		}
	}
}

func entriesToAny(entries []piSessionEntry) []any {
	items := make([]any, 0, len(entries))
	for _, entry := range entries {
		items = append(items, entry)
	}
	return items
}

func strPtr(value string) *string {
	return &value
}
