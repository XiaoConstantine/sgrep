package util

import "strings"

// CombineDescriptionContent merges description and content using the shared
// late-interaction format expected by indexing and search.
func CombineDescriptionContent(content, description string) string {
	description = strings.TrimSpace(description)
	content = strings.TrimSpace(content)

	switch {
	case description == "":
		return content
	case content == "":
		return description
	default:
		return description + "\n\n" + content
	}
}
