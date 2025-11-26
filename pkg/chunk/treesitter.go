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

	description := buildTreeSitterDescription(lang, ntCfg.Kind, name, path)

	return &Chunk{
		Content:     chunkContent,
		StartLine:   int(startPoint.Row) + 1,
		EndLine:     int(endPoint.Row) + 1,
		FilePath:    path,
		Description: description,
	}
}

// buildTreeSitterDescription builds a description for embeddings.
func buildTreeSitterDescription(lang, kind, name, path string) string {
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
