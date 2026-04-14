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

// ClaudeParser parses Claude Code conversation files.
type ClaudeParser struct {
	basePath string
}

// claudeMessage represents a single message in Claude Code JSONL format.
type claudeMessage struct {
	ParentUUID  *string `json:"parentUuid"`
	IsSidechain bool    `json:"isSidechain"`
	UserType    string  `json:"userType"`
	CWD         string  `json:"cwd"`
	SessionID   string  `json:"sessionId"`
	Version     string  `json:"version"`
	GitBranch   string  `json:"gitBranch"`
	GitCommit   string  `json:"gitCommit"`
	Type        string  `json:"type"`
	Message     struct {
		Role    string `json:"role"`
		Content any    `json:"content"` // Can be string or []interface{} (tool use)
	} `json:"message"`
	UUID      string `json:"uuid"`
	Timestamp string `json:"timestamp"`
}

// NewClaudeParser creates a new Claude Code parser.
func NewClaudeParser() *ClaudeParser {
	homeDir, _ := os.UserHomeDir()
	return &ClaudeParser{
		basePath: filepath.Join(homeDir, ".claude"),
	}
}

// NewClaudeParserWithPath creates a parser with a custom base path.
func NewClaudeParserWithPath(basePath string) *ClaudeParser {
	return &ClaudeParser{basePath: basePath}
}

// AgentType returns the agent type.
func (p *ClaudeParser) AgentType() conv.AgentType {
	return conv.AgentClaudeCode
}

// DefaultPath returns the default path for Claude Code conversations.
func (p *ClaudeParser) DefaultPath() string {
	return filepath.Join(p.basePath, "projects")
}

// Discover finds all Claude Code conversation files.
func (p *ClaudeParser) Discover() ([]string, error) {
	var paths []string
	projectsDir := filepath.Join(p.basePath, "projects")

	if _, err := os.Stat(projectsDir); os.IsNotExist(err) {
		return nil, nil
	}

	err := filepath.Walk(projectsDir, func(path string, info os.FileInfo, err error) error {
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

// Parse reads a Claude Code JSONL file and returns sessions.
func (p *ClaudeParser) Parse(sourcePath string) ([]*conv.Session, error) {
	file, err := os.Open(sourcePath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	// Group messages by session ID
	sessionMessages := make(map[string][]claudeMessage)

	err = forEachJSONLLine(file, func(line []byte) error {
		var msg claudeMessage
		if err := json.Unmarshal(line, &msg); err != nil {
			// Skip malformed lines
			return nil
		}

		sessionMessages[msg.SessionID] = append(sessionMessages[msg.SessionID], msg)
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Convert to sessions
	var sessions []*conv.Session
	for sessionID, messages := range sessionMessages {
		session := p.messagesToSession(sessionID, messages, sourcePath)
		if session != nil && len(session.Turns) > 0 {
			sessions = append(sessions, session)
		}
	}

	return sessions, nil
}

// messagesToSession converts Claude messages to a Session.
func (p *ClaudeParser) messagesToSession(sessionID string, messages []claudeMessage, sourcePath string) *conv.Session {
	if len(messages) == 0 {
		return nil
	}

	// Sort messages by timestamp
	sort.Slice(messages, func(i, j int) bool {
		return messages[i].Timestamp < messages[j].Timestamp
	})

	// Extract session metadata from first message
	firstMsg := messages[0]
	session := &conv.Session{
		ID:           sessionID,
		Agent:        conv.AgentClaudeCode,
		AgentVersion: firstMsg.Version,
		SourcePath:   sourcePath,
		ProjectPath:  firstMsg.CWD,
		ProjectName:  filepath.Base(firstMsg.CWD),
		GitBranch:    firstMsg.GitBranch,
		GitCommit:    firstMsg.GitCommit,
	}

	// Parse timestamps
	if ts, err := time.Parse(time.RFC3339, firstMsg.Timestamp); err == nil {
		session.StartedAt = ts
	}
	if len(messages) > 0 {
		lastMsg := messages[len(messages)-1]
		if ts, err := time.Parse(time.RFC3339, lastMsg.Timestamp); err == nil {
			session.EndedAt = ts
		}
	}

	// Group messages into turns (user + assistant pairs)
	var currentTurn *conv.Turn
	turnIndex := 0

	for _, msg := range messages {
		content := extractContent(msg.Message.Content)

		switch msg.Message.Role {
		case "user":
			// Start a new turn
			if currentTurn != nil && currentTurn.UserContent != "" {
				// Previous turn had only user message, save it
				session.Turns = append(session.Turns, *currentTurn)
				turnIndex++
			}
			currentTurn = &conv.Turn{
				Index:       turnIndex,
				UserContent: content,
			}
			if msg.ParentUUID != nil {
				currentTurn.ParentUUID = *msg.ParentUUID
			}
			currentTurn.IsSidechain = msg.IsSidechain
			if ts, err := time.Parse(time.RFC3339, msg.Timestamp); err == nil {
				currentTurn.Timestamp = ts
			}

		case "assistant":
			if currentTurn == nil {
				// Assistant message without user, create empty turn
				currentTurn = &conv.Turn{
					Index: turnIndex,
				}
			}
			currentTurn.AssistContent = content
			currentTurn.HasCode = containsCode(content)
			currentTurn.CodeLangs = detectCodeLanguages(content)

			// Complete the turn
			session.Turns = append(session.Turns, *currentTurn)
			currentTurn = nil
			turnIndex++
		}
	}

	// Save any incomplete turn
	if currentTurn != nil && (currentTurn.UserContent != "" || currentTurn.AssistContent != "") {
		session.Turns = append(session.Turns, *currentTurn)
	}

	return session
}

// extractContent extracts text content from Claude's message content field.
// Claude can have complex content (tool use, etc), but we only want the text.
func extractContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// Content can be an array of content blocks
		var parts []string
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

// containsCode checks if content contains code blocks.
func containsCode(content string) bool {
	return strings.Contains(content, "```")
}

// detectCodeLanguages extracts programming languages from code blocks.
func detectCodeLanguages(content string) []string {
	var langs []string
	seen := make(map[string]bool)

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "```") && len(line) > 3 {
			lang := strings.TrimPrefix(line, "```")
			lang = strings.TrimSpace(lang)
			// Handle common variations
			if idx := strings.IndexAny(lang, " \t{"); idx > 0 {
				lang = lang[:idx]
			}
			if lang != "" && !seen[lang] {
				seen[lang] = true
				langs = append(langs, lang)
			}
		}
	}

	return langs
}

// RegisterClaude registers the Claude parser in the registry.
func RegisterClaude() {
	Register(NewClaudeParser())
}
