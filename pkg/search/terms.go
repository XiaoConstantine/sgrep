package search

import (
	"strings"
)

// Common stopwords to filter out from search queries
var stopwords = map[string]bool{
	// Articles
	"a": true, "an": true, "the": true,
	// Prepositions
	"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
	"with": true, "by": true, "from": true, "as": true,
	// Conjunctions
	"and": true, "or": true, "but": true,
	// Common verbs
	"is": true, "are": true, "was": true, "were": true, "be": true,
	"have": true, "has": true, "had": true, "do": true, "does": true, "did": true,
	// Question words (usually not useful for code search)
	"how": true, "what": true, "where": true, "when": true, "why": true, "which": true,
	// Common search terms
	"find": true, "show": true, "get": true, "list": true,
}

// ExtractSearchTerms converts a natural language query to FTS5 MATCH syntax.
// Returns terms joined with OR for broad matching.
func ExtractSearchTerms(query string) string {
	terms := extractTerms(query)
	if len(terms) == 0 {
		return ""
	}
	// FTS5 OR syntax: term1 OR term2 OR term3
	return strings.Join(terms, " OR ")
}

// ExtractSearchTermsAND converts a natural language query to FTS5 MATCH syntax.
// Returns terms joined with AND for strict matching.
func ExtractSearchTermsAND(query string) string {
	terms := extractTerms(query)
	if len(terms) == 0 {
		return ""
	}
	// FTS5 AND syntax (implicit): term1 term2 term3
	return strings.Join(terms, " ")
}

// extractTerms extracts meaningful search terms from a query.
func extractTerms(query string) []string {
	words := strings.Fields(strings.ToLower(query))
	terms := make([]string, 0, len(words))

	for _, w := range words {
		// Remove punctuation
		w = strings.Trim(w, ".,?!\"'`:;()[]{}*")

		// Skip short words and stopwords
		if len(w) < 2 || stopwords[w] {
			continue
		}

		// Escape special FTS5 characters
		w = escapeFTS5(w)

		if w != "" {
			terms = append(terms, w)
		}
	}

	return terms
}

// escapeFTS5 escapes special characters for FTS5 queries.
func escapeFTS5(term string) string {
	// FTS5 special characters: ^ * " ( ) : -
	// We wrap terms with special chars in double quotes
	needsQuoting := false
	for _, c := range term {
		switch c {
		case '^', '*', '"', '(', ')', ':', '-':
			needsQuoting = true
		}
	}

	if needsQuoting {
		// Escape internal quotes and wrap in quotes
		term = strings.ReplaceAll(term, `"`, `""`)
		return `"` + term + `"`
	}

	return term
}
