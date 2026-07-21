// Package modelcfg centralizes embedding-model and input-budget configuration.
package modelcfg

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	// EmbeddingFormatVersion changes whenever indexed document embeddings become
	// incompatible with query embeddings and therefore require a reindex.
	EmbeddingFormatVersion = 2

	// DefaultContextTokens is the per-slot llama.cpp context budget. Code chunks
	// reserve a small amount of this budget for retrieval task prefixes and model
	// token-estimation variance.
	DefaultContextTokens = 512
	contextReserveTokens = 64
)

// ContextTokens returns the configured per-slot embedding context budget.
func ContextTokens() int {
	if value := strings.TrimSpace(os.Getenv("SGREP_CONTEXT_TOKENS")); value != "" {
		if tokens, err := strconv.Atoi(value); err == nil && tokens >= 128 {
			return tokens
		}
	}
	return DefaultContextTokens
}

// DocumentTokenBudget returns the maximum unprefixed document chunk budget.
func DocumentTokenBudget() int {
	budget := ContextTokens() - contextReserveTokens
	if budget < 96 {
		return 96
	}
	return budget
}

// EstimateTokens conservatively estimates model tokens for source code.
func EstimateTokens(text string) int {
	// Nomic's tokenizer is substantially denser on punctuation-heavy source
	// than natural-language chars/token rules. Two bytes per token leaves enough
	// margin for code without requiring a tokenizer dependency in the CLI.
	charBased := (len(text) + 1) / 2
	wordBased := int(float64(len(strings.Fields(text))) * 1.3)
	if charBased > wordBased {
		return charBased
	}
	return wordBased
}

// ValidateInput rejects oversized model input instead of silently truncating it.
func ValidateInput(text string) error {
	estimated := EstimateTokens(text)
	if estimated > ContextTokens() {
		return fmt.Errorf("embedding input is too large: estimated %d tokens exceeds %d-token slot context", estimated, ContextTokens())
	}
	return nil
}

// QueryText applies Nomic Embed retrieval task formatting.
func QueryText(text string) string {
	return "search_query: " + text
}

// DocumentText applies Nomic Embed retrieval task formatting.
func DocumentText(text string) string {
	return "search_document: " + text
}
