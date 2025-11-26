package chunk

import (
	"unsafe"

	tree_sitter_javascript "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tree_sitter_typescript "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

func init() {
	jsNodeTypes := []NodeTypeConfig{
		{Type: "function_declaration", Kind: "function", NameField: "name", LeadingComment: true},
		{Type: "function_expression", Kind: "function", NameField: "name", LeadingComment: true},
		{Type: "arrow_function", Kind: "arrow function", NameField: "", LeadingComment: true},
		{Type: "class_declaration", Kind: "class", NameField: "name", LeadingComment: true},
		{Type: "method_definition", Kind: "method", NameField: "name", LeadingComment: true},
	}

	RegisterLanguage(&LanguageConfig{
		Name:       "javascript",
		Extensions: []string{".js", ".jsx", ".mjs", ".cjs"},
		Language:   func() unsafe.Pointer { return tree_sitter_javascript.Language() },
		NodeTypes:  jsNodeTypes,
	})

	RegisterLanguage(&LanguageConfig{
		Name:       "typescript",
		Extensions: []string{".ts", ".tsx", ".mts", ".cts"},
		Language:   func() unsafe.Pointer { return tree_sitter_typescript.LanguageTypescript() },
		NodeTypes: append(jsNodeTypes, []NodeTypeConfig{
			{Type: "interface_declaration", Kind: "interface", NameField: "name", LeadingComment: true},
			{Type: "type_alias_declaration", Kind: "type", NameField: "name", LeadingComment: true},
		}...),
	})
}
