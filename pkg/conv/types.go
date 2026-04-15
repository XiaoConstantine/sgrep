// Package conv provides conversation search across coding agents (Claude Code, Codex CLI, Cursor, OpenCode).
package conv

import (
	"time"
)

// AgentType identifies the source coding agent.
type AgentType string

const (
	AgentClaudeCode AgentType = "claude"
	AgentCodexCLI   AgentType = "codex"
	AgentCursor     AgentType = "cursor"
	AgentOpenCode   AgentType = "opencode"
	AgentPiMono     AgentType = "pi"
	AgentAll        AgentType = "all"
)

// ParseAgentType converts a string to AgentType.
func ParseAgentType(s string) AgentType {
	switch s {
	case "claude":
		return AgentClaudeCode
	case "codex":
		return AgentCodexCLI
	case "cursor":
		return AgentCursor
	case "opencode":
		return AgentOpenCode
	case "pi", "pi-mono":
		return AgentPiMono
	default:
		return AgentAll
	}
}

// NormalizeAgentType canonicalizes legacy aliases while preserving unknown values.
func NormalizeAgentType(a AgentType) AgentType {
	switch a {
	case AgentClaudeCode, AgentCodexCLI, AgentCursor, AgentOpenCode, AgentPiMono, AgentAll:
		return a
	case AgentType("pi-mono"):
		return AgentPiMono
	default:
		return a
	}
}

// String returns the string representation of AgentType.
func (a AgentType) String() string {
	return string(a)
}

// Session represents a complete conversation session.
type Session struct {
	// Identity
	ID           string    `json:"id"`
	Agent        AgentType `json:"agent"`
	AgentVersion string    `json:"agent_version,omitempty"`

	// Source
	SourcePath string `json:"source_path"`

	// Project Context
	ProjectPath string `json:"project_path"`
	ProjectName string `json:"project_name"`
	GitBranch   string `json:"git_branch,omitempty"`
	GitCommit   string `json:"git_commit,omitempty"`

	// Timing
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`

	// Content
	Turns []Turn `json:"turns"`

	// Computed
	TotalTokens int `json:"total_tokens,omitempty"`
}

// Turn represents a user-assistant exchange.
type Turn struct {
	Index         int       `json:"index"`
	UserContent   string    `json:"user_content"`
	AssistContent string    `json:"assistant_content"`
	Timestamp     time.Time `json:"timestamp,omitempty"`

	// Code Detection
	HasCode   bool     `json:"has_code,omitempty"`
	CodeLangs []string `json:"code_langs,omitempty"`

	// For Claude Code threading
	ParentUUID  string `json:"parent_uuid,omitempty"`
	IsSidechain bool   `json:"is_sidechain,omitempty"`
}

// ConversationHit represents a search result.
type ConversationHit struct {
	// Session Info
	SessionID   string    `json:"session_id"`
	Agent       AgentType `json:"agent"`
	ProjectPath string    `json:"project_path"`
	ProjectName string    `json:"project_name"`

	// Timing
	StartedAt    time.Time `json:"started_at"`
	RelativeTime string    `json:"relative_time"` // "3 days ago"

	// Matched Content
	MatchedTurn TurnPreview `json:"matched_turn"`

	// Session Metadata
	TotalTurns  int `json:"total_turns"`
	TotalTokens int `json:"total_tokens,omitempty"`

	// Relevance
	Score     float64 `json:"score"`
	MatchType string  `json:"match_type"` // semantic, keyword, hybrid

	// Actionable Commands
	Actions HitActions `json:"actions"`
}

// TurnPreview shows a snippet of the matched turn.
type TurnPreview struct {
	TurnIndex  int      `json:"turn_index"`
	UserSnip   string   `json:"user_snip"`      // Max 80 chars
	AssistSnip string   `json:"assistant_snip"` // Max 100 chars
	Highlights []string `json:"highlights,omitempty"`
	FullUser   string   `json:"full_user,omitempty"`   // Full user content (for verbose output)
	FullAssist string   `json:"full_assist,omitempty"` // Full assistant content (for verbose output)
}

// HitActions contains ready-to-run commands.
type HitActions struct {
	Resume  string `json:"resume"`  // e.g., "claude --resume abc123"
	View    string `json:"view"`    // e.g., "sgrep conv view abc123"
	Export  string `json:"export"`  // e.g., "sgrep conv export abc123"
	Context string `json:"context"` // e.g., "sgrep conv context abc123"
}

// SearchOptions configures conversation search behavior.
type SearchOptions struct {
	Limit          int       `json:"limit"`
	Threshold      float64   `json:"threshold"`
	Agent          AgentType `json:"agent"`
	Project        string    `json:"project"`
	Since          time.Time `json:"since,omitempty"`
	Before         time.Time `json:"before,omitempty"`
	UseHybrid      bool      `json:"use_hybrid"`
	ExactMatch     bool      `json:"exact_match"`
	SemanticWeight float64   `json:"semantic_weight"`
	BM25Weight     float64   `json:"bm25_weight"`
}

// DefaultSearchOptions returns sensible defaults.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		Limit:          10,
		Threshold:      0.5,
		Agent:          AgentAll,
		UseHybrid:      false,
		ExactMatch:     false,
		SemanticWeight: 0.6,
		BM25Weight:     0.4,
	}
}

// SearchResponse represents the full search response.
type SearchResponse struct {
	Query      string            `json:"query"`
	Filters    SearchOptions     `json:"filters"`
	SearchTime int64             `json:"search_time_ms"`
	TotalHits  int               `json:"total_hits"`
	Returned   int               `json:"returned"`
	Results    []ConversationHit `json:"results"`
	Metadata   IndexMetadata     `json:"metadata"`
}

// IndexMetadata contains information about the conversation index.
type IndexMetadata struct {
	IndexVersion    string      `json:"index_version"`
	IndexedSessions int         `json:"indexed_sessions"`
	AgentsIndexed   []AgentType `json:"agents_indexed"`
	LastIndexed     time.Time   `json:"last_indexed"`
}

// ContextOutput represents the output of context extraction.
type ContextOutput struct {
	SessionID     string   `json:"session_id"`
	Agent         string   `json:"agent"`
	Project       string   `json:"project"`
	Date          string   `json:"date"`
	Topic         string   `json:"topic,omitempty"`
	Summary       string   `json:"summary,omitempty"`
	KeyPoints     []string `json:"key_points,omitempty"`
	LastTurns     []Turn   `json:"last_turns"`
	ResumeCommand string   `json:"resume_command"`
}

// IndexStats contains statistics about indexed conversations.
type IndexStats struct {
	TotalSessions   int               `json:"total_sessions"`
	TotalTurns      int               `json:"total_turns"`
	TotalTokens     int               `json:"total_tokens"`
	SessionsByAgent map[AgentType]int `json:"sessions_by_agent"`
	LastIndexed     time.Time         `json:"last_indexed"`
	IndexSizeBytes  int64             `json:"index_size_bytes"`
}

// TurnDocument represents an indexed turn for vector search.
// This is the unit that gets embedded and stored.
type TurnDocument struct {
	ID            string    `json:"id"` // session_id:turn_index
	SessionID     string    `json:"session_id"`
	TurnIndex     int       `json:"turn_index"`
	Content       string    `json:"content"` // Combined user + assistant content
	UserContent   string    `json:"user_content"`
	AssistContent string    `json:"assistant_content"`
	Embedding     []float32 `json:"-"` // Vector embedding
	Agent         AgentType `json:"agent"`
	ProjectPath   string    `json:"project_path"`
	Timestamp     time.Time `json:"timestamp"`
}
