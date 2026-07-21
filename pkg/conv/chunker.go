package conv

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
)

const (
	// MaxTurnTokens documents the default 512-token slot budget after reserve.
	// NewChunker reads the runtime SGREP_CONTEXT_TOKENS setting.
	MaxTurnTokens = 448
	MaxTurnChars  = MaxTurnTokens * 2
	OverlapChars  = 100
)

// Chunker handles turn-based chunking of conversations.
type Chunker struct {
	maxTokens    int
	maxChars     int
	overlapChars int
}

// NewChunker creates a new chunker with default settings.
func NewChunker() *Chunker {
	maxTokens := modelcfg.DocumentTokenBudget()
	return &Chunker{
		maxTokens:    maxTokens,
		maxChars:     maxTokens * 2,
		overlapChars: OverlapChars,
	}
}

// ChunkerConfig allows customizing chunker behavior.
type ChunkerConfig struct {
	MaxTokens    int
	OverlapChars int
}

// NewChunkerWithConfig creates a chunker with custom configuration.
func NewChunkerWithConfig(cfg ChunkerConfig) *Chunker {
	maxChars := cfg.MaxTokens * 2
	if maxChars == 0 {
		maxChars = MaxTurnChars
	}
	overlapChars := cfg.OverlapChars
	if overlapChars == 0 {
		overlapChars = OverlapChars
	}
	return &Chunker{
		maxTokens:    maxChars / 2,
		maxChars:     maxChars,
		overlapChars: overlapChars,
	}
}

// TurnChunk represents a chunk derived from a turn.
type TurnChunk struct {
	ID            string // session_id:turn_index[:chunk_index]
	SessionID     string
	TurnIndex     int
	ChunkIndex    int    // 0 for single chunk, 1+ for split chunks
	Content       string // Combined user + assistant content
	UserContent   string // Original user content
	AssistContent string // Original assistant content (may be partial if split)
}

// ChunkTurn converts a turn into one or more chunks.
// Most turns fit in a single chunk; long turns are split at paragraph boundaries.
func (c *Chunker) ChunkTurn(sessionID string, turn *Turn) []TurnChunk {
	// Create combined content for embedding
	combinedContent := c.formatTurnContent(turn)

	// maxChars is a conservative byte budget (two bytes per model token).
	if modelcfg.EstimateTokens(combinedContent) <= c.maxTokens {
		return []TurnChunk{{
			ID:            fmt.Sprintf("%s:%d", sessionID, turn.Index),
			SessionID:     sessionID,
			TurnIndex:     turn.Index,
			ChunkIndex:    0,
			Content:       combinedContent,
			UserContent:   turn.UserContent,
			AssistContent: turn.AssistContent,
		}}
	}

	// Need to split - split the assistant content (usually the long part)
	return c.splitTurn(sessionID, turn)
}

// ChunkSession processes an entire session into chunks.
func (c *Chunker) ChunkSession(session *Session) []TurnChunk {
	var chunks []TurnChunk
	for i := range session.Turns {
		turnChunks := c.ChunkTurn(session.ID, &session.Turns[i])
		chunks = append(chunks, turnChunks...)
	}
	return chunks
}

// formatTurnContent creates the combined content for embedding.
func (c *Chunker) formatTurnContent(turn *Turn) string {
	var sb strings.Builder
	sb.WriteString("USER: ")
	sb.WriteString(strings.TrimSpace(turn.UserContent))
	sb.WriteString("\n\nASSISTANT: ")
	sb.WriteString(strings.TrimSpace(turn.AssistContent))
	return sb.String()
}

// splitTurn splits the complete formatted turn. Splitting only the assistant
// side fails when the user prompt itself consumes the model context budget.
func (c *Chunker) splitTurn(sessionID string, turn *Turn) []TurnChunk {
	parts := c.splitAtParagraphs(c.formatTurnContent(turn), c.maxChars)
	budgeted := make([]string, 0, len(parts))
	for _, part := range parts {
		if modelcfg.EstimateTokens(part) > c.maxTokens {
			budgeted = append(budgeted, c.hardSplitByTokens(part)...)
		} else {
			budgeted = append(budgeted, part)
		}
	}
	chunks := make([]TurnChunk, 0, len(budgeted))
	for i, part := range budgeted {
		chunks = append(chunks, TurnChunk{
			ID:            fmt.Sprintf("%s:%d:%d", sessionID, turn.Index, i),
			SessionID:     sessionID,
			TurnIndex:     turn.Index,
			ChunkIndex:    i,
			Content:       strings.TrimSpace(part),
			UserContent:   turn.UserContent,
			AssistContent: turn.AssistContent,
		})
	}
	return chunks
}

// splitAtParagraphs splits text at paragraph boundaries (double newlines).
func (c *Chunker) splitAtParagraphs(text string, maxChars int) []string {
	if maxChars <= 0 {
		maxChars = c.maxChars
	}

	text = strings.TrimSpace(text)
	if len(text) <= maxChars {
		return []string{text}
	}

	var parts []string
	paragraphs := strings.Split(text, "\n\n")

	var current strings.Builder
	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		paraLen := len(para)
		currentLen := current.Len()

		// If adding this paragraph would exceed limit
		if currentLen > 0 && currentLen+paraLen+2 > maxChars {
			// Save current and start new
			parts = append(parts, current.String())
			current.Reset()

			// Add overlap from end of previous content
			if c.overlapChars > 0 && len(parts) > 0 {
				prev := parts[len(parts)-1]
				runes := []rune(prev)
				if len(runes) > c.overlapChars {
					overlap := string(runes[len(runes)-c.overlapChars:])
					// Find last complete sentence or line
					if idx := strings.LastIndex(overlap, ". "); idx > 0 {
						overlap = overlap[idx+2:]
					} else if idx := strings.LastIndex(overlap, "\n"); idx > 0 {
						overlap = overlap[idx+1:]
					}
					current.WriteString(overlap)
					current.WriteString("\n\n")
				}
			}
		}

		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(para)
	}

	// Don't forget the last part
	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	// Handle case where a single paragraph is too long
	var finalParts []string
	for _, part := range parts {
		if len(part) > maxChars {
			// Split by sentences or hard break
			subParts := c.hardSplit(part, maxChars)
			finalParts = append(finalParts, subParts...)
		} else {
			finalParts = append(finalParts, part)
		}
	}

	return finalParts
}

// hardSplit splits text when no good boundaries exist.
func (c *Chunker) hardSplit(text string, maxChars int) []string {
	var parts []string
	remaining := text
	for len(remaining) > 0 {
		end := min(maxChars, len(remaining))
		for end > 0 && end < len(remaining) && !utf8.RuneStart(remaining[end]) {
			end--
		}
		if end == 0 {
			_, size := utf8.DecodeRuneInString(remaining)
			end = size
		}
		if end < len(remaining) {
			candidate := remaining[:end]
			boundary := -1
			for _, delim := range []string{". ", "! ", "? ", ".\n", "!\n", "?\n", " "} {
				if idx := strings.LastIndex(candidate, delim); idx > end/2 {
					boundary = idx + len(delim)
					break
				}
			}
			if boundary > 0 {
				end = boundary
			}
		}
		part := strings.TrimSpace(remaining[:end])
		if part != "" {
			parts = append(parts, part)
		}
		remaining = remaining[end:]
		if len(remaining) > 0 && c.overlapChars > 0 && len(parts) > 0 {
			overlapBytes := min(c.overlapChars, maxChars/4)
			previous := parts[len(parts)-1]
			start := max(0, len(previous)-overlapBytes)
			for start < len(previous) && !utf8.RuneStart(previous[start]) {
				start++
			}
			remaining = previous[start:] + remaining
		}
	}
	return parts
}

func (c *Chunker) hardSplitByTokens(text string) []string {
	var parts []string
	remaining := []rune(strings.TrimSpace(text))
	for len(remaining) > 0 {
		low, high := 1, len(remaining)
		for low < high {
			mid := (low + high + 1) / 2
			if modelcfg.EstimateTokens(string(remaining[:mid])) <= c.maxTokens {
				low = mid
			} else {
				high = mid - 1
			}
		}
		end := low
		if end < len(remaining) {
			candidate := string(remaining[:end])
			if boundary := strings.LastIndexAny(candidate, ".!?\n "); boundary > len(candidate)/2 {
				end = utf8.RuneCountInString(candidate[:boundary+1])
			}
		}
		part := strings.TrimSpace(string(remaining[:end]))
		if part != "" {
			parts = append(parts, part)
		}
		remaining = remaining[end:]
		if len(remaining) > 0 && c.overlapChars > 0 && len(parts) > 0 {
			previous := []rune(parts[len(parts)-1])
			overlap := min(c.overlapChars, max(1, end/4))
			if overlap < len(previous) {
				remaining = append(append([]rune(nil), previous[len(previous)-overlap:]...), remaining...)
			}
		}
	}
	return parts
}

// EstimateTokens estimates the token count for text.
// Uses a simple heuristic of ~4 characters per token.
func EstimateTokens(text string) int {
	return (utf8.RuneCountInString(text) + 3) / 4
}

// EstimateTurnTokens estimates tokens for a turn.
func EstimateTurnTokens(turn *Turn) int {
	return EstimateTokens(turn.UserContent) + EstimateTokens(turn.AssistContent)
}

// EstimateSessionTokens estimates total tokens for a session.
func EstimateSessionTokens(session *Session) int {
	total := 0
	for _, turn := range session.Turns {
		total += EstimateTurnTokens(&turn)
	}
	return total
}
