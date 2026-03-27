package conv

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// Actions handles conversation actions like resume, export, and context extraction.
type Actions struct {
	store *Store
}

// NewActions creates a new Actions handler.
func NewActions(store *Store) *Actions {
	return &Actions{store: store}
}

// ViewOptions configures view output.
type ViewOptions struct {
	Turn    int  // Specific turn to view (-1 for all)
	JSONOut bool // Output as JSON
	Verbose bool // Show full content
}

// View returns a session for viewing.
func (a *Actions) View(ctx context.Context, sessionID string, opts ViewOptions) (*Session, error) {
	session, err := a.store.GetSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("session not found: %w", err)
	}

	// If specific turn requested, filter
	if opts.Turn >= 0 && opts.Turn < len(session.Turns) {
		session.Turns = []Turn{session.Turns[opts.Turn]}
	}

	return session, nil
}

// ResumeOptions configures resume behavior.
type ResumeOptions struct {
	FromTurn int  // Resume from specific turn (-1 for full session)
	DryRun   bool // Show command without executing
}

// ResumeResult contains the result of a resume operation.
type ResumeResult struct {
	Command     string   // The resume command
	Args        []string // Command arguments
	SessionID   string
	Agent       AgentType
	ProjectPath string
	Executed    bool
	Error       error
}

// Resume generates or executes a resume command.
func (a *Actions) Resume(ctx context.Context, sessionID string, opts ResumeOptions) (*ResumeResult, error) {
	session, err := a.store.GetSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("session not found: %w", err)
	}

	result := &ResumeResult{
		SessionID:   sessionID,
		Agent:       session.Agent,
		ProjectPath: session.ProjectPath,
	}

	// Generate command based on agent
	switch session.Agent {
	case AgentClaudeCode:
		result.Command = "claude"
		result.Args = []string{"--resume", sessionID}
		if opts.FromTurn > 0 {
			// Claude Code doesn't support --from, but we can document it
			result.Args = append(result.Args, fmt.Sprintf("--continue-from=%d", opts.FromTurn))
		}

	case AgentCodexCLI:
		result.Command = "codex"
		result.Args = []string{"resume", sessionID}

	case AgentCursor:
		// Cursor doesn't have CLI resume
		result.Command = "cursor"
		result.Args = []string{session.ProjectPath}
		// This will just open the project; user needs to find the chat manually
	case AgentOpenCode:
		result.Command = "opencode"
		result.Args = []string{"--session", sessionID}

	default:
		return nil, fmt.Errorf("unsupported agent for resume: %s", session.Agent)
	}

	// Execute if not dry run
	if !opts.DryRun {
		cmd := exec.Command(result.Command, result.Args...)
		cmd.Dir = session.ProjectPath
		if err := cmd.Start(); err != nil {
			result.Error = err
		} else {
			result.Executed = true
		}
	}

	return result, nil
}

// ContextOptions configures context extraction.
type ContextOptions struct {
	Turns   int    // Number of recent turns to include (default: 5)
	Summary bool   // Generate condensed summary
	Format  string // Output format: prompt, markdown, json
	Copy    bool   // Copy to clipboard
}

// ContextResult contains extracted context.
type ContextResult struct {
	SessionID       string    `json:"session_id"`
	Agent           AgentType `json:"agent"`
	Project         string    `json:"project"`
	Date            time.Time `json:"date"`
	Topic           string    `json:"topic,omitempty"`
	Summary         string    `json:"summary,omitempty"`
	KeyPoints       []string  `json:"key_points,omitempty"`
	LastTurns       []Turn    `json:"last_turns"`
	ResumeCommand   string    `json:"resume_command"`
	FormattedOutput string    `json:"-"` // For CLI display
}

// ExtractContext extracts context from a session for injection into new sessions.
func (a *Actions) ExtractContext(ctx context.Context, sessionID string, opts ContextOptions) (*ContextResult, error) {
	session, err := a.store.GetSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("session not found: %w", err)
	}

	// Default to 5 turns
	if opts.Turns <= 0 {
		opts.Turns = 5
	}

	// Get last N turns
	startIdx := len(session.Turns) - opts.Turns
	if startIdx < 0 {
		startIdx = 0
	}
	lastTurns := session.Turns[startIdx:]

	result := &ContextResult{
		SessionID:     sessionID,
		Agent:         session.Agent,
		Project:       session.ProjectName,
		Date:          session.StartedAt,
		LastTurns:     lastTurns,
		ResumeCommand: GenerateResumeCommand(session),
	}

	// Generate topic from first user message
	if len(session.Turns) > 0 {
		result.Topic = extractTopic(session.Turns[0].UserContent)
	}

	// Generate summary if requested
	if opts.Summary {
		result.Summary = generateSummary(session)
		result.KeyPoints = extractKeyPoints(session)
	}

	// Format output
	if opts.Format == "" {
		opts.Format = "prompt"
	}
	result.FormattedOutput = formatContextOutput(result, opts.Format)

	// Copy to clipboard if requested
	if opts.Copy {
		_ = copyToClipboard(result.FormattedOutput)
	}

	return result, nil
}

// ExportOptions configures export behavior.
type ExportOptions struct {
	Format string // markdown, json, html
	Output string // Output file path (empty for stdout)
}

// Export exports a session to a file.
func (a *Actions) Export(ctx context.Context, sessionID string, opts ExportOptions) (string, error) {
	session, err := a.store.GetSession(ctx, sessionID)
	if err != nil {
		return "", fmt.Errorf("session not found: %w", err)
	}

	if opts.Format == "" {
		opts.Format = "markdown"
	}

	switch opts.Format {
	case "markdown":
		return exportMarkdown(session), nil
	case "json":
		return exportJSON(session)
	case "html":
		return exportHTML(session), nil
	default:
		return "", fmt.Errorf("unsupported format: %s", opts.Format)
	}
}

// CopyOptions configures copy behavior.
type CopyOptions struct {
	Turn     int  // Specific turn (-1 for all)
	CodeOnly bool // Copy only code blocks
	Full     bool // Copy full conversation
}

// Copy copies conversation content to clipboard.
func (a *Actions) Copy(ctx context.Context, sessionID string, opts CopyOptions) (string, error) {
	session, err := a.store.GetSession(ctx, sessionID)
	if err != nil {
		return "", fmt.Errorf("session not found: %w", err)
	}

	var content string

	if opts.Turn >= 0 && opts.Turn < len(session.Turns) {
		// Copy specific turn
		turn := session.Turns[opts.Turn]
		if opts.CodeOnly {
			content = extractCodeBlocks(turn.AssistContent)
		} else {
			content = fmt.Sprintf("USER: %s\n\nASSISTANT: %s", turn.UserContent, turn.AssistContent)
		}
	} else if opts.Full {
		// Copy full conversation
		content = exportMarkdown(session)
	} else {
		// Copy code from all turns
		var codes []string
		for _, turn := range session.Turns {
			if code := extractCodeBlocks(turn.AssistContent); code != "" {
				codes = append(codes, code)
			}
		}
		content = strings.Join(codes, "\n\n---\n\n")
	}

	if err := copyToClipboard(content); err != nil {
		return content, fmt.Errorf("failed to copy to clipboard: %w", err)
	}

	return content, nil
}

// Helper functions

// GenerateResumeCommand generates the agent-specific resume command for a session.
func GenerateResumeCommand(session *Session) string {
	switch session.Agent {
	case AgentClaudeCode:
		return fmt.Sprintf("claude --resume %s", session.ID)
	case AgentCodexCLI:
		return fmt.Sprintf("codex resume %s", session.ID)
	case AgentCursor:
		return fmt.Sprintf("# Open Cursor: %s", session.ProjectPath)
	case AgentOpenCode:
		return fmt.Sprintf("opencode --session %s", session.ID)
	default:
		return ""
	}
}

func extractTopic(firstUserMessage string) string {
	// Take first line or first 50 chars
	lines := strings.SplitN(firstUserMessage, "\n", 2)
	topic := lines[0]
	if len(topic) > 50 {
		if idx := strings.LastIndex(topic[:50], " "); idx > 20 {
			topic = topic[:idx] + "..."
		} else {
			topic = topic[:47] + "..."
		}
	}
	return topic
}

func generateSummary(session *Session) string {
	if len(session.Turns) == 0 {
		return ""
	}

	// Simple summary: first user query + indication of length
	firstQuery := extractTopic(session.Turns[0].UserContent)
	return fmt.Sprintf("Discussion about: %s (%d turns)", firstQuery, len(session.Turns))
}

func extractKeyPoints(session *Session) []string {
	var points []string

	// Extract key points from user messages
	for i, turn := range session.Turns {
		if i >= 5 {
			break // Limit to first 5 turns
		}
		point := extractTopic(turn.UserContent)
		if point != "" {
			points = append(points, fmt.Sprintf("Turn %d: %s", i+1, point))
		}
	}

	return points
}

func formatContextOutput(result *ContextResult, format string) string {
	var sb strings.Builder

	switch format {
	case "markdown":
		sb.WriteString("## Previous Session Context\n\n")
		fmt.Fprintf(&sb, "**Project:** %s\n", result.Project)
		fmt.Fprintf(&sb, "**Date:** %s (%s)\n", result.Date.Format("2006-01-02"), relativeTime(result.Date))
		fmt.Fprintf(&sb, "**Agent:** %s\n", result.Agent)
		if result.Topic != "" {
			fmt.Fprintf(&sb, "**Topic:** %s\n", result.Topic)
		}
		sb.WriteString("\n")

		if result.Summary != "" {
			fmt.Fprintf(&sb, "### Summary\n%s\n\n", result.Summary)
		}

		if len(result.KeyPoints) > 0 {
			sb.WriteString("### Key Decisions\n")
			for _, point := range result.KeyPoints {
				sb.WriteString(fmt.Sprintf("- %s\n", point))
			}
			sb.WriteString("\n")
		}

		for i, turn := range result.LastTurns {
			sb.WriteString(fmt.Sprintf("### Turn %d\n", i+1))
			sb.WriteString(fmt.Sprintf("**USER:** %s\n\n", turn.UserContent))
			sb.WriteString(fmt.Sprintf("**ASSISTANT:** %s\n\n", turn.AssistContent))
		}

		sb.WriteString("---\n")
		sb.WriteString(fmt.Sprintf("*Resume: %s*\n", result.ResumeCommand))

	case "prompt":
		sb.WriteString("## Previous Session Context\n\n")
		sb.WriteString(fmt.Sprintf("**Project:** %s\n", result.Project))
		sb.WriteString(fmt.Sprintf("**Date:** %s (%s)\n", result.Date.Format("2006-01-02"), relativeTime(result.Date)))
		sb.WriteString(fmt.Sprintf("**Agent:** %s\n", result.Agent))
		if result.Topic != "" {
			sb.WriteString(fmt.Sprintf("**Topic:** %s\n\n", result.Topic))
		}

		// Show last turns in condensed format
		for i, turn := range result.LastTurns {
			sb.WriteString(fmt.Sprintf("### Exchange %d\n", i+1))
			sb.WriteString(fmt.Sprintf("USER: %s\n\n", turn.UserContent))
			// Truncate long assistant responses
			assistContent := turn.AssistContent
			if len(assistContent) > 500 {
				assistContent = assistContent[:497] + "..."
			}
			sb.WriteString(fmt.Sprintf("ASSISTANT: %s\n\n", assistContent))
		}

		sb.WriteString("---\n")
		sb.WriteString(fmt.Sprintf("*Continue from this context. Session: %s*\n", result.SessionID))

	default: // json is handled elsewhere
		sb.WriteString(fmt.Sprintf("Session: %s\nProject: %s\n", result.SessionID, result.Project))
	}

	return sb.String()
}

func exportMarkdown(session *Session) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("# Conversation: %s\n\n", session.ProjectName))
	sb.WriteString(fmt.Sprintf("**Agent:** %s\n", session.Agent))
	sb.WriteString(fmt.Sprintf("**Project:** %s\n", session.ProjectPath))
	sb.WriteString(fmt.Sprintf("**Date:** %s\n", session.StartedAt.Format("2006-01-02 15:04:05")))
	if session.GitBranch != "" {
		sb.WriteString(fmt.Sprintf("**Branch:** %s\n", session.GitBranch))
	}
	sb.WriteString("\n---\n\n")

	for i, turn := range session.Turns {
		sb.WriteString(fmt.Sprintf("## Turn %d\n\n", i+1))
		sb.WriteString(fmt.Sprintf("### User\n\n%s\n\n", turn.UserContent))
		sb.WriteString(fmt.Sprintf("### Assistant\n\n%s\n\n", turn.AssistContent))
		sb.WriteString("---\n\n")
	}

	return sb.String()
}

func exportJSON(session *Session) (string, error) {
	// Use standard JSON encoding
	data, err := json.Marshal(session)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func exportHTML(session *Session) string {
	var sb strings.Builder

	sb.WriteString("<!DOCTYPE html>\n<html>\n<head>\n")
	sb.WriteString("<meta charset=\"UTF-8\">\n")
	sb.WriteString(fmt.Sprintf("<title>Conversation: %s</title>\n", session.ProjectName))
	sb.WriteString("<style>\n")
	sb.WriteString("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n")
	sb.WriteString(".turn { margin-bottom: 20px; }\n")
	sb.WriteString(".user { background: #e3f2fd; padding: 15px; border-radius: 8px; }\n")
	sb.WriteString(".assistant { background: #f5f5f5; padding: 15px; border-radius: 8px; }\n")
	sb.WriteString("pre { background: #263238; color: #fff; padding: 10px; border-radius: 4px; overflow-x: auto; }\n")
	sb.WriteString("</style>\n</head>\n<body>\n")

	sb.WriteString(fmt.Sprintf("<h1>%s</h1>\n", session.ProjectName))
	sb.WriteString(fmt.Sprintf("<p><strong>Agent:</strong> %s | <strong>Date:</strong> %s</p>\n",
		session.Agent, session.StartedAt.Format("2006-01-02")))
	sb.WriteString("<hr>\n")

	for i, turn := range session.Turns {
		sb.WriteString("<div class=\"turn\">\n")
		sb.WriteString(fmt.Sprintf("<h3>Turn %d</h3>\n", i+1))
		sb.WriteString(fmt.Sprintf("<div class=\"user\"><strong>User:</strong><br>%s</div>\n",
			escapeHTML(turn.UserContent)))
		sb.WriteString(fmt.Sprintf("<div class=\"assistant\"><strong>Assistant:</strong><br>%s</div>\n",
			formatHTMLContent(turn.AssistContent)))
		sb.WriteString("</div>\n")
	}

	sb.WriteString("</body>\n</html>")
	return sb.String()
}

func escapeHTML(s string) string {
	escaped := html.EscapeString(s)
	return strings.ReplaceAll(escaped, "\n", "<br>")
}

func formatHTMLContent(s string) string {
	// Basic markdown code block to HTML
	lines := strings.Split(s, "\n")
	var result []string
	inCode := false

	for _, line := range lines {
		if strings.HasPrefix(line, "```") {
			if inCode {
				result = append(result, "</pre>")
				inCode = false
			} else {
				result = append(result, "<pre>")
				inCode = true
			}
			continue
		}

		// Inside code blocks, escape but preserve newlines as actual newlines
		// Outside code blocks, escape and convert newlines to <br>
		if inCode {
			result = append(result, html.EscapeString(line))
		} else {
			result = append(result, html.EscapeString(line)+"<br>")
		}
	}

	if inCode {
		result = append(result, "</pre>")
	}

	return strings.Join(result, "\n")
}

func extractCodeBlocks(content string) string {
	var codes []string
	lines := strings.Split(content, "\n")
	var currentCode strings.Builder
	inCode := false

	for _, line := range lines {
		if strings.HasPrefix(line, "```") {
			if inCode {
				if currentCode.Len() > 0 {
					codes = append(codes, currentCode.String())
				}
				currentCode.Reset()
				inCode = false
			} else {
				inCode = true
			}
			continue
		}

		if inCode {
			if currentCode.Len() > 0 {
				currentCode.WriteString("\n")
			}
			currentCode.WriteString(line)
		}
	}

	return strings.Join(codes, "\n\n")
}

func copyToClipboard(content string) error {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("pbcopy")
	case "linux":
		// Try xclip first, then xsel
		if _, err := exec.LookPath("xclip"); err == nil {
			cmd = exec.Command("xclip", "-selection", "clipboard")
		} else if _, err := exec.LookPath("xsel"); err == nil {
			cmd = exec.Command("xsel", "--clipboard", "--input")
		} else {
			return fmt.Errorf("no clipboard utility found (install xclip or xsel)")
		}
	case "windows":
		cmd = exec.Command("clip")
	default:
		return fmt.Errorf("clipboard not supported on %s", runtime.GOOS)
	}

	cmd.Stdin = strings.NewReader(content)
	return cmd.Run()
}
