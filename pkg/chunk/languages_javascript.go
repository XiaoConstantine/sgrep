package chunk

import (
	"path/filepath"
	"strings"
	"unsafe"

	tree_sitter_javascript "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tree_sitter_typescript "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

func init() {
	jsNodeTypes := []NodeTypeConfig{
		{Type: "function_declaration", Kind: "function", NameField: "name", LeadingComment: true},
		{Type: "generator_function_declaration", Kind: "generator function", NameField: "name", LeadingComment: true},
		{Type: "function_expression", Kind: "function", NameField: "name", LeadingComment: true, InferNameFromParent: true},
		{Type: "generator_function", Kind: "generator function", NameField: "name", LeadingComment: true, InferNameFromParent: true},
		{Type: "arrow_function", Kind: "arrow function", LeadingComment: true, InferNameFromParent: true},
		{Type: "class_declaration", Kind: "class", NameField: "name", LeadingComment: true},
		{Type: "method_definition", Kind: "method", NameField: "name", LeadingComment: true},
	}

	RegisterLanguage(&LanguageConfig{
		Name:       "javascript",
		Extensions: []string{".js", ".jsx", ".mjs", ".cjs"},
		Language:   func(string) unsafe.Pointer { return tree_sitter_javascript.Language() },
		NodeTypes:  jsNodeTypes,
	})

	RegisterLanguage(&LanguageConfig{
		Name:       "typescript",
		Extensions: []string{".ts", ".tsx", ".mts", ".cts"},
		Language: func(path string) unsafe.Pointer {
			if strings.EqualFold(filepath.Ext(path), ".tsx") {
				return tree_sitter_typescript.LanguageTSX()
			}
			return tree_sitter_typescript.LanguageTypescript()
		},
		NodeTypes: append(jsNodeTypes, []NodeTypeConfig{
			{Type: "abstract_class_declaration", Kind: "class", NameField: "name", LeadingComment: true},
			{Type: "function_signature", Kind: "function", NameField: "name", LeadingComment: true},
			{Type: "interface_declaration", Kind: "interface", NameField: "name", LeadingComment: true},
			{Type: "type_alias_declaration", Kind: "type", NameField: "name", LeadingComment: true},
			{Type: "enum_declaration", Kind: "enum", NameField: "name", LeadingComment: true},
			{Type: "internal_module", Kind: "namespace", NameField: "name", LeadingComment: true},
		}...),
	})
}
