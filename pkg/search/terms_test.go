package search

import (
	"testing"
)

func TestExtractSearchTerms(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		expected string
	}{
		{
			name:     "simple query",
			query:    "authentication middleware",
			expected: "authentication OR middleware",
		},
		{
			name:     "query with stopwords",
			query:    "how does the authentication work",
			expected: "authentication OR work",
		},
		{
			name:     "query with punctuation",
			query:    "error handling, logging, and metrics",
			expected: "error OR handling OR logging OR metrics",
		},
		{
			name:     "single word",
			query:    "authentication",
			expected: "authentication",
		},
		{
			name:     "all stopwords",
			query:    "how is the",
			expected: "",
		},
		{
			name:     "empty query",
			query:    "",
			expected: "",
		},
		{
			name:     "short words filtered",
			query:    "a b c authentication",
			expected: "authentication",
		},
		{
			name:     "mixed case normalized",
			query:    "Authentication MIDDLEWARE Handler",
			expected: "authentication OR middleware OR handler",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractSearchTerms(tt.query)
			if result != tt.expected {
				t.Errorf("ExtractSearchTerms(%q) = %q, want %q", tt.query, result, tt.expected)
			}
		})
	}
}

func TestExtractSearchTermsAND(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		expected string
	}{
		{
			name:     "simple query",
			query:    "authentication middleware",
			expected: "authentication middleware",
		},
		{
			name:     "query with stopwords",
			query:    "how does the authentication work",
			expected: "authentication work",
		},
		{
			name:     "empty after filtering",
			query:    "the a an",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractSearchTermsAND(tt.query)
			if result != tt.expected {
				t.Errorf("ExtractSearchTermsAND(%q) = %q, want %q", tt.query, result, tt.expected)
			}
		})
	}
}

func TestEscapeFTS5(t *testing.T) {
	tests := []struct {
		name     string
		term     string
		expected string
	}{
		{
			name:     "no special chars",
			term:     "authentication",
			expected: "authentication",
		},
		{
			name:     "with hyphen",
			term:     "rate-limit",
			expected: `"rate-limit"`,
		},
		{
			name:     "with colon",
			term:     "http:error",
			expected: `"http:error"`,
		},
		{
			name:     "with asterisk",
			term:     "auth*",
			expected: `"auth*"`,
		},
		{
			name:     "with parentheses",
			term:     "func()",
			expected: `"func()"`,
		},
		{
			name:     "with quotes",
			term:     `say"hello"`,
			expected: `"say""hello"""`,
		},
		{
			name:     "with caret",
			term:     "^start",
			expected: `"^start"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := escapeFTS5(tt.term)
			if result != tt.expected {
				t.Errorf("escapeFTS5(%q) = %q, want %q", tt.term, result, tt.expected)
			}
		})
	}
}

func TestExtractTerms(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		expected []string
	}{
		{
			name:     "basic extraction",
			query:    "error handling in database",
			expected: []string{"error", "handling", "database"},
		},
		{
			name:     "filters stopwords",
			query:    "the quick brown fox",
			expected: []string{"quick", "brown", "fox"},
		},
		{
			name:     "removes punctuation",
			query:    "hello, world! how are you?",
			expected: []string{"hello", "world", "you"},
		},
		{
			name:     "handles special FTS5 chars",
			query:    "rate-limit handler",
			expected: []string{`"rate-limit"`, "handler"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractTerms(tt.query)
			if len(result) != len(tt.expected) {
				t.Errorf("extractTerms(%q) returned %d terms, want %d", tt.query, len(result), len(tt.expected))
				t.Errorf("got: %v, want: %v", result, tt.expected)
				return
			}
			for i, term := range result {
				if term != tt.expected[i] {
					t.Errorf("extractTerms(%q)[%d] = %q, want %q", tt.query, i, term, tt.expected[i])
				}
			}
		})
	}
}
