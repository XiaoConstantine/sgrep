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

// PiMonoParser parses pi-mono coding-agent session files.
type PiMonoParser struct {
	basePath string
}

type piSessionHeader struct {
	Type      string `json:"type"`
	Version   int    `json:"version,omitempty"`
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`
	CWD       string `json:"cwd"`
}

type piMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type piSessionEntry struct {
	Type      string     `json:"type"`
	ID        string     `json:"id"`
	ParentID  *string    `json:"parentId"`
	Timestamp string     `json:"timestamp"`
	Message   *piMessage `json:"message,omitempty"`
}

type piTurnMessage struct {
	role      string
	content   string
	timestamp time.Time
}

// NewPiMonoParser creates a new pi-mono parser.
func NewPiMonoParser() *PiMonoParser {
	homeDir, _ := os.UserHomeDir()
	return &PiMonoParser{
		basePath: filepath.Join(homeDir, ".pi", "agent"),
	}
}

// NewPiMonoParserWithPath creates a parser with a custom base path.
func NewPiMonoParserWithPath(basePath string) *PiMonoParser {
	return &PiMonoParser{basePath: basePath}
}

// AgentType returns the agent type.
func (p *PiMonoParser) AgentType() conv.AgentType {
	return conv.AgentPiMono
}

// DefaultPath returns the default path for pi-mono sessions.
func (p *PiMonoParser) DefaultPath() string {
	return filepath.Join(p.basePath, "sessions")
}

// Discover finds all pi-mono session files.
func (p *PiMonoParser) Discover() ([]string, error) {
	var paths []string
	sessionsDir := p.DefaultPath()

	if _, err := os.Stat(sessionsDir); os.IsNotExist(err) {
		return nil, nil
	}

	err := filepath.Walk(sessionsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(path, ".jsonl") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Strings(paths)
	return paths, nil
}

// Parse reads a pi-mono session file and returns a session.
func (p *PiMonoParser) Parse(sourcePath string) ([]*conv.Session, error) {
	file, err := os.Open(sourcePath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	var header *piSessionHeader
	var entries []piSessionEntry

	err = forEachJSONLLine(file, func(line []byte) error {
		var kind struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(line, &kind); err != nil {
			return nil
		}

		switch kind.Type {
		case "session":
			var parsed piSessionHeader
			if err := json.Unmarshal(line, &parsed); err == nil && parsed.ID != "" {
				header = &parsed
			}
		default:
			var parsed piSessionEntry
			if err := json.Unmarshal(line, &parsed); err == nil && parsed.ID != "" {
				entries = append(entries, parsed)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if header == nil {
		return nil, nil
	}

	messages, endedAt := extractPiLeafMessages(entries)
	if len(messages) == 0 {
		return nil, nil
	}

	startedAt, _ := parsePiTime(header.Timestamp)
	session := &conv.Session{
		ID:          header.ID,
		Agent:       conv.AgentPiMono,
		SourcePath:  sourcePath,
		ProjectPath: header.CWD,
		ProjectName: filepath.Base(header.CWD),
		StartedAt:   startedAt,
		EndedAt:     endedAt,
		Turns:       piMessagesToTurns(messages),
	}
	if session.StartedAt.IsZero() && len(session.Turns) > 0 {
		session.StartedAt = session.Turns[0].Timestamp
	}
	if session.EndedAt.IsZero() && len(session.Turns) > 0 {
		session.EndedAt = session.Turns[len(session.Turns)-1].Timestamp
	}
	if len(session.Turns) == 0 {
		return nil, nil
	}

	return []*conv.Session{session}, nil
}

func extractPiLeafMessages(entries []piSessionEntry) ([]piTurnMessage, time.Time) {
	if len(entries) == 0 {
		return nil, time.Time{}
	}

	byID := make(map[string]piSessionEntry, len(entries))
	leafID := ""
	for _, entry := range entries {
		byID[entry.ID] = entry
		leafID = entry.ID
	}
	if leafID == "" {
		return nil, time.Time{}
	}

	var path []piSessionEntry
	currentID := leafID
	for currentID != "" {
		entry, ok := byID[currentID]
		if !ok {
			break
		}
		path = append(path, entry)
		if entry.ParentID == nil || *entry.ParentID == "" {
			break
		}
		currentID = *entry.ParentID
	}

	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	var messages []piTurnMessage
	var endedAt time.Time
	for _, entry := range path {
		ts, _ := parsePiTime(entry.Timestamp)
		if !ts.IsZero() {
			endedAt = ts
		}
		if entry.Type != "message" || entry.Message == nil {
			continue
		}

		content := extractPiMessageContent(entry.Message.Content)
		if content == "" {
			continue
		}

		switch entry.Message.Role {
		case "user", "assistant":
			messages = append(messages, piTurnMessage{
				role:      entry.Message.Role,
				content:   content,
				timestamp: ts,
			})
		}
	}

	return messages, endedAt
}

func piMessagesToTurns(messages []piTurnMessage) []conv.Turn {
	var turns []conv.Turn
	var currentTurn *conv.Turn
	turnIndex := 0

	for _, msg := range messages {
		switch msg.role {
		case "user":
			if currentTurn != nil && currentTurn.UserContent != "" {
				turns = append(turns, *currentTurn)
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
			if currentTurn.Timestamp.IsZero() {
				currentTurn.Timestamp = msg.timestamp
			}
			turns = append(turns, *currentTurn)
			currentTurn = nil
			turnIndex++
		}
	}

	if currentTurn != nil && (currentTurn.UserContent != "" || currentTurn.AssistContent != "") {
		turns = append(turns, *currentTurn)
	}

	return turns
}

func extractPiMessageContent(content any) string {
	switch v := content.(type) {
	case string:
		return strings.TrimSpace(v)
	case []interface{}:
		var parts []string
		for _, item := range v {
			block, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			text := extractPiTextBlock(block)
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func extractPiTextBlock(block map[string]interface{}) string {
	blockType, _ := block["type"].(string)
	if blockType != "text" {
		return ""
	}
	text, _ := block["text"].(string)
	return strings.TrimSpace(text)
}

func parsePiTime(value string) (time.Time, bool) {
	if value == "" {
		return time.Time{}, false
	}
	if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return ts, true
	}
	if ts, err := time.Parse(time.RFC3339, value); err == nil {
		return ts, true
	}
	return time.Time{}, false
}

// RegisterPiMono registers the pi-mono parser in the registry.
func RegisterPiMono() {
	Register(NewPiMonoParser())
}
