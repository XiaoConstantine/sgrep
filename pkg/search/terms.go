package search

import (
	"strings"
	"unicode"
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

// ExtractHybridSearchTerms converts a natural language query to FTS5 MATCH syntax
// with code-identifier compound expansions for hybrid lexical retrieval.
func ExtractHybridSearchTerms(query string) string {
	terms := extractTerms(query)
	if len(terms) == 0 {
		return ""
	}
	if len(terms) == 1 {
		return terms[0]
	}
	// Candidate generation should favor recall. Each term group retains exact
	// compound identifier alternatives, while groups are unioned and later
	// fused with dense retrieval using weighted RRF.
	return strings.Join(groupedCompoundTerms(terms), " OR ")
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

func groupedCompoundTerms(terms []string) []string {
	compounds := compoundTermsByInputIndex(terms)
	groups := make([]string, 0, len(terms))
	for i, term := range terms {
		alternatives := make([]string, 0, 1+len(compounds[i]))
		seen := make(map[string]struct{}, 1+len(compounds[i]))
		add := func(value string) {
			if value == "" {
				return
			}
			if _, ok := seen[value]; ok {
				return
			}
			seen[value] = struct{}{}
			alternatives = append(alternatives, value)
		}
		add(term)
		for _, compound := range compounds[i] {
			add(compound)
		}
		if len(alternatives) == 1 {
			groups = append(groups, alternatives[0])
			continue
		}
		groups = append(groups, "("+strings.Join(alternatives, " OR ")+")")
	}
	return groups
}

func compoundTermsByInputIndex(terms []string) map[int][]string {
	compounds := make(map[int][]string, len(terms))
	const maxNgram = 4
	for width := 2; width <= maxNgram && width <= len(terms); width++ {
		for start := 0; start+width <= len(terms); start++ {
			var b strings.Builder
			skip := false
			for _, term := range terms[start : start+width] {
				if strings.Contains(term, `"`) {
					skip = true
					break
				}
				b.WriteString(term)
			}
			if skip {
				continue
			}
			compound := b.String()
			for i := start; i < start+width; i++ {
				compounds[i] = append(compounds[i], compound)
			}
		}
	}
	return compounds
}

// escapeFTS5 escapes special characters for FTS5 queries.
func escapeFTS5(term string) string {
	// FTS5 barewords permit letters, numbers, underscore, and non-ASCII
	// letters. Quote everything else (not just operators) so paths and file
	// extensions cannot turn into invalid MATCH syntax.
	needsQuoting := term == ""
	for _, c := range term {
		if c != '_' && !unicode.IsLetter(c) && !unicode.IsNumber(c) {
			needsQuoting = true
			break
		}
	}

	if needsQuoting {
		// Escape internal quotes and wrap in quotes
		term = strings.ReplaceAll(term, `"`, `""`)
		return `"` + term + `"`
	}

	return term
}
