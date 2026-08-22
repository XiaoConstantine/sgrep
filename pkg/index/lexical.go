package index

import (
	"path/filepath"
	"regexp"
	"strings"
)

var (
	lexicalIdentifierPattern = regexp.MustCompile(`[A-Za-z][A-Za-z0-9_-]*`)
	camelAcronymBoundary     = regexp.MustCompile(`([A-Z]{2,})([A-Z][a-z])`)
	camelWordBoundary        = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	lexicalSeparatorReplacer = strings.NewReplacer("_", " ", "-", " ")
)

// buildLexicalText supplements raw source FTS with AST descriptions, path
// components, and identifier pieces that unicode61 does not split itself.
func buildLexicalText(path, description, content string) string {
	input := description + " " + filepath.ToSlash(path) + " " + content
	identifiers := lexicalIdentifierPattern.FindAllString(input, -1)
	seen := make(map[string]struct{}, len(identifiers)*2)
	terms := make([]string, 0, len(identifiers)*2)
	add := func(term string) {
		term = strings.ToLower(strings.TrimSpace(term))
		if len(term) < 2 {
			return
		}
		if _, ok := seen[term]; ok {
			return
		}
		seen[term] = struct{}{}
		terms = append(terms, term)
	}

	for _, identifier := range identifiers {
		add(identifier)
		expanded := lexicalSeparatorReplacer.Replace(identifier)
		expanded = camelAcronymBoundary.ReplaceAllString(expanded, `$1 $2`)
		expanded = camelWordBoundary.ReplaceAllString(expanded, `$1 $2`)
		for _, part := range strings.Fields(expanded) {
			add(part)
		}
	}
	return strings.Join(terms, " ")
}
