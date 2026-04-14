package parser

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
)

// CodexParser parses OpenAI Codex CLI conversation files.
type CodexParser struct {
	basePath string
}

// codexEntry represents a single entry in Codex CLI JSONL format.
type codexEntry struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"` // session_meta, message, etc.
	Payload   json.RawMessage `json:"payload"`
}

// codexSessionMeta represents session metadata.
type codexSessionMeta struct {
	ID            string `json:"id"`
	CWD           string `json:"cwd"`
	ModelProvider string `json:"model_provider"`
	Git           struct {
		Branch     string `json:"branch"`
		CommitHash string `json:"commit_hash"`
	} `json:"git"`
}

// codexMessage represents a message payload.
type codexMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type codexMessageContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type codexResponseMessage struct {
	Type    string                `json:"type"`
	Role    string                `json:"role"`
	Content []codexMessageContent `json:"content"`
}

// NewCodexParser creates a new Codex CLI parser.
func NewCodexParser() *CodexParser {
	homeDir, _ := os.UserHomeDir()
	return &CodexParser{
		basePath: filepath.Join(homeDir, ".codex"),
	}
}

// NewCodexParserWithPath creates a parser with a custom base path.
func NewCodexParserWithPath(basePath string) *CodexParser {
	return &CodexParser{basePath: basePath}
}

// AgentType returns the agent type.
func (p *CodexParser) AgentType() conv.AgentType {
	return conv.AgentCodexCLI
}

// DefaultPath returns the default path for Codex CLI conversations.
func (p *CodexParser) DefaultPath() string {
	return filepath.Join(p.basePath, "sessions")
}

// Discover finds all Codex CLI conversation files.
func (p *CodexParser) Discover() ([]string, error) {
	var paths []string
	sessionsDir := filepath.Join(p.basePath, "sessions")

	if _, err := os.Stat(sessionsDir); os.IsNotExist(err) {
		return nil, nil
	}

	// Walk through YYYY/MM/DD directory structure
	err := filepath.Walk(sessionsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(path, ".jsonl") {
			paths = append(paths, path)
		}
		return nil
	})

	return paths, err
}

// Parse reads a Codex CLI JSONL file and returns sessions.
func (p *CodexParser) Parse(sourcePath string) ([]*conv.Session, error) {
	file, err := os.Open(sourcePath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	var entries []codexEntry
	err = forEachJSONLLine(file, func(line []byte) error {
		var entry codexEntry
		if err := json.Unmarshal(line, &entry); err == nil {
			entries = append(entries, entry)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Convert to session
	session := p.entriesToSession(entries, sourcePath)
	if session == nil || len(session.Turns) == 0 {
		return nil, nil
	}

	return []*conv.Session{session}, nil
}

// entriesToSession converts Codex entries to a Session.
func (p *CodexParser) entriesToSession(entries []codexEntry, sourcePath string) *conv.Session {
	if len(entries) == 0 {
		return nil
	}

	session := &conv.Session{
		Agent:      conv.AgentCodexCLI,
		SourcePath: sourcePath,
	}

	var messages []struct {
		role      string
		content   string
		timestamp time.Time
	}

	for _, entry := range entries {
		ts, _ := time.Parse(time.RFC3339, entry.Timestamp)

		switch entry.Type {
		case "session_meta":
			var meta codexSessionMeta
			if err := json.Unmarshal(entry.Payload, &meta); err == nil {
				session.ID = meta.ID
				session.ProjectPath = meta.CWD
				session.ProjectName = filepath.Base(meta.CWD)
				session.GitBranch = meta.Git.Branch
				session.GitCommit = meta.Git.CommitHash
				session.AgentVersion = meta.ModelProvider
				session.StartedAt = ts
			}

		case "message":
			var msg codexMessage
			if err := json.Unmarshal(entry.Payload, &msg); err == nil {
				messages = append(messages, struct {
					role      string
					content   string
					timestamp time.Time
				}{msg.Role, msg.Content, ts})
			}
		case "response_item":
			var msg codexResponseMessage
			if err := json.Unmarshal(entry.Payload, &msg); err == nil && msg.Type == "message" {
				content := joinCodexContent(msg.Content)
				if content != "" {
					messages = append(messages, struct {
						role      string
						content   string
						timestamp time.Time
					}{msg.Role, content, ts})
				}
			}
		}
	}

	// Generate session ID from source path if not set
	if session.ID == "" {
		session.ID = filepath.Base(sourcePath)
		session.ID = strings.TrimSuffix(session.ID, ".jsonl")
	}

	// Sort messages by timestamp
	sort.Slice(messages, func(i, j int) bool {
		return messages[i].timestamp.Before(messages[j].timestamp)
	})

	// Group messages into turns
	var currentTurn *conv.Turn
	turnIndex := 0

	for _, msg := range messages {
		switch msg.role {
		case "user":
			if currentTurn != nil && currentTurn.UserContent != "" {
				session.Turns = append(session.Turns, *currentTurn)
				turnIndex++
			}
			currentTurn = &conv.Turn{
				Index:       turnIndex,
				UserContent: msg.content,
				Timestamp:   msg.timestamp,
			}

		case "assistant":
			if currentTurn == nil {
				currentTurn = &conv.Turn{
					Index:     turnIndex,
					Timestamp: msg.timestamp,
				}
			}
			currentTurn.AssistContent = msg.content
			currentTurn.HasCode = containsCode(msg.content)
			currentTurn.CodeLangs = detectCodeLanguages(msg.content)

			session.Turns = append(session.Turns, *currentTurn)
			currentTurn = nil
			turnIndex++
		}
	}

	// Save any incomplete turn
	if currentTurn != nil && currentTurn.UserContent != "" {
		session.Turns = append(session.Turns, *currentTurn)
	}

	// Set end time from last message
	if len(messages) > 0 {
		session.EndedAt = messages[len(messages)-1].timestamp
	}

	return session
}

func joinCodexContent(parts []codexMessageContent) string {
	var sb strings.Builder
	for _, part := range parts {
		if part.Text == "" {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(part.Text)
	}
	return sb.String()
}

// RegisterCodex registers the Codex parser in the registry.
func RegisterCodex() {
	Register(NewCodexParser())
}
