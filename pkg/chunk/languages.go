package chunk

import (
	"path/filepath"
	"strings"
	"unsafe"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
)

// LanguageConfig defines how to parse and chunk a language.
type LanguageConfig struct {
	Name       string
	Extensions []string
	Language   func() unsafe.Pointer // Returns tree-sitter language pointer
	NodeTypes  []NodeTypeConfig      // Semantic units to extract
}

// NodeTypeConfig defines a semantic unit to extract from the AST.
type NodeTypeConfig struct {
	Type           string // tree-sitter node type (e.g., "function_definition")
	Kind           string // Human-readable kind (e.g., "function", "class")
	NameField      string // Field name for the identifier (e.g., "name")
	DocstringField string // Field name for docstring (e.g., "body" for Python where first child may be string)
	DocstringType  string // Node type for docstring (e.g., "expression_statement" containing "string")
	LeadingComment bool   // If true, look for doc comments before the node (JS/TS JSDoc, Rust ///)
}

// Registry holds all supported languages.
var Registry = map[string]*LanguageConfig{}

// extensionMap for fast lookup by file extension.
var extensionMap = map[string]*LanguageConfig{}

// RegisterLanguage adds a language to the registry.
func RegisterLanguage(cfg *LanguageConfig) {
	Registry[cfg.Name] = cfg
	for _, ext := range cfg.Extensions {
		extensionMap[ext] = cfg
	}
}

// GetLanguageByPath returns the language config for a file path.
func GetLanguageByPath(path string) *LanguageConfig {
	ext := strings.ToLower(filepath.Ext(path))
	return extensionMap[ext]
}

// GetLanguageByExt returns the language config for a file extension.
func GetLanguageByExt(ext string) *LanguageConfig {
	return extensionMap[strings.ToLower(ext)]
}

// NewLanguage creates a tree-sitter Language from an unsafe pointer.
func NewLanguage(ptr unsafe.Pointer) *tree_sitter.Language {
	return tree_sitter.NewLanguage(ptr)
}
