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

	lang := NewLanguage(langCfg.Language(path))
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
	if root.HasError() {
		return chunkBySize(path, content, cfg)
	}

	// Walk tree and extract semantic units
	var ancestorNames [8]string
	walkTree(root, []byte(content), path, langCfg.Name, langCfg.nodeTypes, cfg, ancestorNames[:0], nil, false, &chunks)

	if len(chunks) == 0 {
		return chunkBySize(path, content, cfg)
	}

	return splitOversizedChunks(chunks, cfg), nil
}

// walkTree recursively walks the AST and extracts chunks.
func walkTree(node *tree_sitter.Node, content []byte, path, lang string,
	nodeTypes map[string]NodeTypeConfig, cfg *Config, ancestorNames []string,
	commentAnchor *tree_sitter.Node, skipCurrent bool, chunks *[]Chunk) {

	nodeType := node.Kind()
	if isTransparentDeclarationWrapper(nodeType) && commentAnchor == nil {
		commentAnchor = node
	}
	ntCfg, configured := nodeTypes[nodeType]
	var semanticNode *tree_sitter.Node
	var semanticCfg NodeTypeConfig
	semantic := false
	if configured {
		if ntCfg.UnwrapField != "" || ntCfg.UnwrapChild {
			semanticNode, semanticCfg, semantic = resolveSemanticNode(node, ntCfg, nodeTypes)
		} else {
			semanticNode, semanticCfg, semantic = node, ntCfg, ntCfg.Kind != ""
		}
	}

	localName := ""
	if semantic && !skipCurrent {
		localName = extractNodeName(semanticNode, content, semanticCfg)
	}
	if configured && semantic && !skipCurrent {
		chunk := extractChunk(
			node, semanticNode, commentAnchor, content, path, lang, localName, ancestorNames, semanticCfg,
		)
		// Lower threshold for tree-sitter since semantic units are meaningful even if small
		if chunk != nil && estimateTokens(chunk.Content) >= 10 {
			*chunks = append(*chunks, *chunk)
		}
	}

	var wrappedChild *tree_sitter.Node
	if configured && (ntCfg.UnwrapField != "" || ntCfg.UnwrapChild) {
		wrappedChild = directWrappedChild(node, ntCfg, nodeTypes)
	}
	childAncestorNames := ancestorNames
	if semantic && localName != "" && !skipCurrent {
		childAncestorNames = append(childAncestorNames, localName)
	}
	childCommentAnchor := commentAnchor
	if semantic && !skipCurrent {
		childCommentAnchor = nil
	}

	// Recurse into children
	childCount := node.NamedChildCount()
	for i := uint(0); i < uint(childCount); i++ {
		child := node.NamedChild(uint(i))
		if child != nil {
			skipChild := wrappedChild != nil && wrappedChild.Id() == child.Id()
			walkTree(
				child, content, path, lang, nodeTypes, cfg, childAncestorNames, childCommentAnchor, skipChild, chunks,
			)
		}
	}
}

func isTransparentDeclarationWrapper(nodeType string) bool {
	return nodeType == "export_statement" || nodeType == "ambient_declaration"
}

func resolveSemanticNode(node *tree_sitter.Node, cfg NodeTypeConfig,
	nodeTypes map[string]NodeTypeConfig) (*tree_sitter.Node, NodeTypeConfig, bool) {
	current := node
	currentCfg := cfg

	for currentCfg.UnwrapField != "" || currentCfg.UnwrapChild {
		child := directWrappedChild(current, currentCfg, nodeTypes)
		if child == nil {
			return nil, NodeTypeConfig{}, false
		}
		childCfg, ok := nodeTypes[child.Kind()]
		if !ok {
			return nil, NodeTypeConfig{}, false
		}
		current = child
		currentCfg = childCfg
	}

	return current, currentCfg, currentCfg.Kind != ""
}

func directWrappedChild(node *tree_sitter.Node, cfg NodeTypeConfig,
	nodeTypes map[string]NodeTypeConfig) *tree_sitter.Node {
	if cfg.UnwrapField != "" {
		return node.ChildByFieldName(cfg.UnwrapField)
	}
	if !cfg.UnwrapChild {
		return nil
	}

	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child != nil {
			if _, ok := nodeTypes[child.Kind()]; ok {
				return child
			}
		}
	}
	return nil
}

// extractChunk extracts a chunk from a node.
func extractChunk(node, semanticNode, commentAnchor *tree_sitter.Node, content []byte,
	path, lang, name string, ancestorNames []string, ntCfg NodeTypeConfig) *Chunk {
	rangeNode := semanticRangeNode(node, semanticNode, ntCfg)
	startByte := rangeNode.StartByte()
	endByte := rangeNode.EndByte()
	startPoint := rangeNode.StartPosition()
	endPoint := rangeNode.EndPosition()

	if startByte >= uint(len(content)) || endByte > uint(len(content)) {
		return nil
	}

	chunkContent := string(content[startByte:endByte])

	// Extract docstring if available
	docstring := ""
	if ntCfg.DocstringField != "" && ntCfg.DocstringType != "" {
		// Python-style: docstring inside body
		docstring = extractDocstring(semanticNode, content, ntCfg)
	} else if ntCfg.LeadingComment {
		anchor := rangeNode
		if commentAnchor != nil {
			anchor = commentAnchor
		}
		var commentStart uint
		var commentLine int
		var ok bool
		docstring, commentStart, commentLine, ok = extractLeadingComment(
			anchor, content, lang, startByte, startPoint.Row,
		)
		if ok {
			startByte = commentStart
			startPoint.Row = uint(commentLine - 1)
			chunkContent = string(content[startByte:endByte])
		}
	}

	description := buildTreeSitterDescription(lang, ntCfg.Kind, ancestorNames, name, path, docstring)

	return &Chunk{
		Content:     chunkContent,
		StartLine:   int(startPoint.Row) + 1,
		EndLine:     int(endPoint.Row) + 1,
		FilePath:    path,
		Description: description,
	}
}

func semanticRangeNode(node, semanticNode *tree_sitter.Node, cfg NodeTypeConfig) *tree_sitter.Node {
	if !cfg.InferNameFromParent || node.Id() != semanticNode.Id() {
		return node
	}

	parent := semanticNode.Parent()
	for parent != nil {
		switch parent.Kind() {
		case "parenthesized_expression", "as_expression", "satisfies_expression",
			"type_assertion", "non_null_expression":
			parent = parent.Parent()
			continue
		case "variable_declarator":
			declaration := parent.Parent()
			if declaration != nil && (declaration.Kind() == "lexical_declaration" ||
				declaration.Kind() == "variable_declaration") &&
				countNamedChildren(declaration, "variable_declarator") == 1 {
				return declaration
			}
			return parent
		case "pair", "assignment_expression", "public_field_definition", "field_definition":
			return parent
		default:
			return node
		}
	}
	return node
}

func countNamedChildren(node *tree_sitter.Node, kind string) int {
	count := 0
	for i := uint(0); i < node.NamedChildCount(); i++ {
		child := node.NamedChild(i)
		if child != nil && child.Kind() == kind {
			count++
		}
	}
	return count
}

func extractNodeName(node *tree_sitter.Node, content []byte, cfg NodeTypeConfig) string {
	var name string
	if cfg.NameField != "" {
		nameNode := node.ChildByFieldName(cfg.NameField)
		if cfg.NameField == "declarator" {
			nameNode = innermostDeclarator(nameNode)
		}
		name = nodeText(nameNode, content)
	}
	if name == "" && cfg.InferNameFromParent {
		name = inferAssignedName(node, content)
	}
	return strings.TrimSpace(name)
}

func innermostDeclarator(node *tree_sitter.Node) *tree_sitter.Node {
	for node != nil {
		child := node.ChildByFieldName("declarator")
		if child == nil {
			return node
		}
		node = child
	}
	return nil
}

func inferAssignedName(node *tree_sitter.Node, content []byte) string {
	parent := node.Parent()
	for parent != nil {
		var fields []string
		switch parent.Kind() {
		case "variable_declarator", "public_field_definition", "field_definition":
			fields = []string{"name", "property"}
		case "pair":
			fields = []string{"key"}
		case "assignment_expression":
			fields = []string{"left"}
		case "parenthesized_expression", "as_expression", "satisfies_expression",
			"type_assertion", "non_null_expression":
			parent = parent.Parent()
			continue
		default:
			return ""
		}

		for _, field := range fields {
			if name := nodeText(parent.ChildByFieldName(field), content); name != "" {
				return name
			}
		}
		return ""
	}
	return ""
}

func nodeText(node *tree_sitter.Node, content []byte) string {
	if node == nil {
		return ""
	}
	startByte, endByte := node.StartByte(), node.EndByte()
	if startByte >= uint(len(content)) || endByte > uint(len(content)) {
		return ""
	}
	return strings.TrimSpace(string(content[startByte:endByte]))
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

// extractLeadingComment extracts consecutive documentation comments directly before a node.
func extractLeadingComment(node *tree_sitter.Node, content []byte, lang string,
	nodeStartByte, nodeStartRow uint) (string, uint, int, bool) {
	var comments []string
	startByte := nodeStartByte
	startLine := int(nodeStartRow) + 1
	nextRow := nodeStartRow

	for sibling := node.PrevNamedSibling(); sibling != nil; sibling = sibling.PrevNamedSibling() {
		if sibling.StartByte() >= uint(len(content)) || sibling.EndByte() > uint(len(content)) {
			break
		}
		rawComment := string(content[sibling.StartByte():sibling.EndByte()])
		endsWithNewline := strings.HasSuffix(rawComment, "\n")
		if (endsWithNewline && sibling.EndPosition().Row != nextRow) ||
			(!endsWithNewline && sibling.EndPosition().Row+1 != nextRow) {
			break
		}
		comment := strings.TrimSpace(rawComment)
		if !isDocumentationComment(sibling.Kind(), comment, lang) {
			break
		}

		comments = append([]string{cleanComment(comment)}, comments...)
		startByte = sibling.StartByte()
		startLine = int(sibling.StartPosition().Row) + 1
		nextRow = sibling.StartPosition().Row
	}

	if len(comments) == 0 {
		return "", 0, 0, false
	}
	return limitSummary(strings.Join(comments, " "), 200), startByte, startLine, true
}

func limitSummary(summary string, maxLength int) string {
	if len(summary) <= maxLength {
		return summary
	}
	if boundary := strings.LastIndexByte(summary[:maxLength], ' '); boundary > 0 {
		return summary[:boundary]
	}
	return summary[:maxLength]
}

func isDocumentationComment(kind, comment, lang string) bool {
	switch kind {
	case "comment", "line_comment", "block_comment":
	default:
		return false
	}

	comment = strings.TrimSpace(comment)
	switch lang {
	case "rust":
		return strings.HasPrefix(comment, "///") || strings.HasPrefix(comment, "//!") ||
			strings.HasPrefix(comment, "/**") || strings.HasPrefix(comment, "/*!")
	case "c", "cpp":
		return strings.HasPrefix(comment, "/**") || strings.HasPrefix(comment, "/*!") ||
			strings.HasPrefix(comment, "///") || strings.HasPrefix(comment, "//!")
	default:
		return strings.HasPrefix(comment, "/**") || strings.HasPrefix(comment, "///")
	}
}

// cleanComment removes comment markers and cleans up.
func cleanComment(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "/**")
	s = strings.TrimPrefix(s, "/*!")
	s = strings.TrimPrefix(s, "/*")
	s = strings.TrimSuffix(s, "*/")

	// Clean up JSDoc annotations and asterisks
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		line = strings.TrimPrefix(line, "///")
		line = strings.TrimPrefix(line, "//!")
		line = strings.TrimPrefix(line, "//")
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
func buildTreeSitterDescription(lang, kind string, ancestorNames []string, name, path, docstring string) string {
	baseName := filepath.Base(path)
	descriptionLen := len(lang) + 1 + len(kind) + len(" in ") + len(baseName)
	if name != "" {
		descriptionLen += 1 + len(name)
		for _, ancestorName := range ancestorNames {
			descriptionLen += len(ancestorName) + 1
		}
	}
	if docstring != "" {
		descriptionLen += len(". ") + len(docstring)
	}

	var b strings.Builder
	b.Grow(descriptionLen)
	b.WriteString(strings.ToUpper(lang[:1]))
	b.WriteString(lang[1:])
	b.WriteString(" ")
	b.WriteString(kind)
	if name != "" {
		b.WriteString(" ")
		for _, ancestorName := range ancestorNames {
			b.WriteString(ancestorName)
			b.WriteByte('.')
		}
		b.WriteString(name)
	}
	b.WriteString(" in ")
	b.WriteString(baseName)

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
