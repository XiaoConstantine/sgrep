package chunk

import (
	"path/filepath"
	"strings"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
)

// chunkTreeSitter uses tree-sitter to extract semantic chunks.
func chunkTreeSitter(path string, content string, cfg *Config, langCfg *LanguageConfig) ([]Chunk, error) {
	parser := tree_sitter.NewParser()
	defer parser.Close()

	lang := NewLanguage(langCfg.Language())
	if err := parser.SetLanguage(lang); err != nil {
		return chunkBySize(path, content, cfg)
	}

	tree := parser.Parse([]byte(content), nil)
	if tree == nil {
		return chunkBySize(path, content, cfg)
	}
	defer tree.Close()

	var chunks []Chunk
	root := tree.RootNode()

	// Build set of node types to extract
	nodeTypeSet := make(map[string]NodeTypeConfig)
	for _, nt := range langCfg.NodeTypes {
		nodeTypeSet[nt.Type] = nt
	}

	// Walk tree and extract semantic units
	walkTree(root, []byte(content), path, langCfg.Name, nodeTypeSet, cfg, &chunks)

	if len(chunks) == 0 {
		return chunkBySize(path, content, cfg)
	}

	return splitOversizedChunks(chunks, cfg), nil
}

// walkTree recursively walks the AST and extracts chunks.
func walkTree(node *tree_sitter.Node, content []byte, path, lang string,
	nodeTypes map[string]NodeTypeConfig, cfg *Config, chunks *[]Chunk) {

	nodeType := node.Kind()

	if ntCfg, ok := nodeTypes[nodeType]; ok {
		chunk := extractChunk(node, content, path, lang, ntCfg)
		// Lower threshold for tree-sitter since semantic units are meaningful even if small
		if chunk != nil && estimateTokens(chunk.Content) >= 10 {
			*chunks = append(*chunks, *chunk)
		}
	}

	// Recurse into children
	childCount := node.NamedChildCount()
	for i := uint(0); i < uint(childCount); i++ {
		child := node.NamedChild(uint(i))
		if child != nil {
			walkTree(child, content, path, lang, nodeTypes, cfg, chunks)
		}
	}
}

// extractChunk extracts a chunk from a node.
func extractChunk(node *tree_sitter.Node, content []byte, path, lang string, ntCfg NodeTypeConfig) *Chunk {
	startByte := node.StartByte()
	endByte := node.EndByte()
	startPoint := node.StartPosition()
	endPoint := node.EndPosition()

	if startByte >= uint(len(content)) || endByte > uint(len(content)) {
		return nil
	}

	chunkContent := string(content[startByte:endByte])

	// Extract name if available
	name := ""
	if ntCfg.NameField != "" {
		if nameNode := node.ChildByFieldName(ntCfg.NameField); nameNode != nil {
			nameStart := nameNode.StartByte()
			nameEnd := nameNode.EndByte()
			if nameStart < uint(len(content)) && nameEnd <= uint(len(content)) {
				name = string(content[nameStart:nameEnd])
			}
		}
	}

	// Extract docstring if available
	docstring := ""
	if ntCfg.DocstringField != "" && ntCfg.DocstringType != "" {
		// Python-style: docstring inside body
		docstring = extractDocstring(node, content, ntCfg)
	} else if ntCfg.LeadingComment {
		// JS/Rust-style: doc comment before the node
		docstring = extractLeadingComment(node, content)
	}

	description := buildTreeSitterDescription(lang, ntCfg.Kind, name, path, docstring)

	return &Chunk{
		Content:     chunkContent,
		StartLine:   int(startPoint.Row) + 1,
		EndLine:     int(endPoint.Row) + 1,
		FilePath:    path,
		Description: description,
	}
}

// extractDocstring extracts a docstring from a function/class body.
func extractDocstring(node *tree_sitter.Node, content []byte, ntCfg NodeTypeConfig) string {
	// Get the body field
	bodyNode := node.ChildByFieldName(ntCfg.DocstringField)
	if bodyNode == nil {
		return ""
	}

	// Look for the first child that matches docstring type (e.g., expression_statement)
	childCount := bodyNode.NamedChildCount()
	if childCount == 0 {
		return ""
	}

	firstChild := bodyNode.NamedChild(0)
	if firstChild == nil || firstChild.Kind() != ntCfg.DocstringType {
		return ""
	}

	// For Python, expression_statement contains a string node
	// For JS/TS, we might look for a comment node
	stringNode := firstChild.NamedChild(0)
	if stringNode == nil {
		return ""
	}

	// Check if it's a string literal (Python docstring)
	kind := stringNode.Kind()
	if kind != "string" && kind != "concatenated_string" {
		return ""
	}

	startByte := stringNode.StartByte()
	endByte := stringNode.EndByte()
	if startByte >= uint(len(content)) || endByte > uint(len(content)) {
		return ""
	}

	docstring := string(content[startByte:endByte])

	// Clean up the docstring - remove quotes and trim
	docstring = cleanDocstring(docstring)

	return docstring
}

// extractLeadingComment extracts JSDoc or Rust doc comments before a node.
func extractLeadingComment(node *tree_sitter.Node, content []byte) string {
	// Get the previous sibling - doc comments appear right before the function
	prevSibling := node.PrevSibling()
	if prevSibling == nil {
		return ""
	}

	kind := prevSibling.Kind()

	// Check for comment types
	isDocComment := false
	switch kind {
	case "comment":
		// JS/TS: Could be JSDoc /** ... */ or regular comment
		isDocComment = true
	case "line_comment":
		// Rust: /// doc comment
		isDocComment = true
	case "block_comment":
		// Rust: /** ... */ or /* ... */
		isDocComment = true
	}

	if !isDocComment {
		return ""
	}

	startByte := prevSibling.StartByte()
	endByte := prevSibling.EndByte()
	if startByte >= uint(len(content)) || endByte > uint(len(content)) {
		return ""
	}

	comment := string(content[startByte:endByte])

	// Clean up the comment
	return cleanComment(comment)
}

// cleanComment removes comment markers and cleans up.
func cleanComment(s string) string {
	// Remove JSDoc style /** ... */
	if strings.HasPrefix(s, "/**") {
		s = strings.TrimPrefix(s, "/**")
		s = strings.TrimSuffix(s, "*/")
	} else if strings.HasPrefix(s, "/*") {
		s = strings.TrimPrefix(s, "/*")
		s = strings.TrimSuffix(s, "*/")
	}

	// Remove Rust doc comment ///
	if strings.HasPrefix(s, "///") {
		s = strings.TrimPrefix(s, "///")
	} else if strings.HasPrefix(s, "//!") {
		s = strings.TrimPrefix(s, "//!")
	} else if strings.HasPrefix(s, "//") {
		s = strings.TrimPrefix(s, "//")
	}

	// Clean up JSDoc annotations and asterisks
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		// Remove leading asterisks from JSDoc
		line = strings.TrimPrefix(line, "* ")
		line = strings.TrimPrefix(line, "*")
		line = strings.TrimSpace(line)

		// Skip @param, @returns, etc. - keep only description
		if strings.HasPrefix(line, "@") {
			continue
		}
		if line != "" {
			cleaned = append(cleaned, line)
		}

		// Limit to ~200 chars
		if len(strings.Join(cleaned, " ")) > 200 {
			break
		}
	}

	return strings.Join(cleaned, " ")
}

// cleanDocstring removes quotes and cleans up a docstring.
func cleanDocstring(s string) string {
	// Remove triple quotes (""" or ''')
	s = strings.TrimPrefix(s, `"""`)
	s = strings.TrimSuffix(s, `"""`)
	s = strings.TrimPrefix(s, `'''`)
	s = strings.TrimSuffix(s, `'''`)
	// Remove single/double quotes
	s = strings.TrimPrefix(s, `"`)
	s = strings.TrimSuffix(s, `"`)
	s = strings.TrimPrefix(s, `'`)
	s = strings.TrimSuffix(s, `'`)

	// Trim whitespace and get first line/paragraph
	s = strings.TrimSpace(s)

	// Get first paragraph (up to blank line or 200 chars)
	lines := strings.Split(s, "\n")
	var result []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" && len(result) > 0 {
			break // Stop at first blank line after content
		}
		if line != "" {
			result = append(result, line)
		}
		if len(strings.Join(result, " ")) > 200 {
			break
		}
	}

	return strings.Join(result, " ")
}

// buildTreeSitterDescription builds a description for embeddings.
func buildTreeSitterDescription(lang, kind, name, path, docstring string) string {
	var b strings.Builder
	b.WriteString(strings.ToUpper(lang[:1]))
	b.WriteString(lang[1:])
	b.WriteString(" ")
	b.WriteString(kind)
	if name != "" {
		b.WriteString(" ")
		b.WriteString(name)
	}
	b.WriteString(" in ")
	b.WriteString(filepath.Base(path))

	// Append docstring if available
	if docstring != "" {
		b.WriteString(". ")
		b.WriteString(docstring)
	}

	return b.String()
}

// splitOversizedChunks splits chunks that exceed MaxTokens.
func splitOversizedChunks(chunks []Chunk, cfg *Config) []Chunk {
	var result []Chunk
	for _, chunk := range chunks {
		if estimateTokens(chunk.Content) > cfg.MaxTokens {
			result = append(result, splitOversized(chunk, cfg)...)
		} else {
			result = append(result, chunk)
		}
	}
	return result
}
