package chunk

import (
	"unsafe"

	tree_sitter_javascript "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tree_sitter_typescript "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

func init() {
	jsNodeTypes := []NodeTypeConfig{
		{Type: "function_declaration", Kind: "function", NameField: "name"},
		{Type: "function_expression", Kind: "function", NameField: "name"},
		{Type: "arrow_function", Kind: "arrow function", NameField: ""},
		{Type: "class_declaration", Kind: "class", NameField: "name"},
		{Type: "method_definition", Kind: "method", NameField: "name"},
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
			{Type: "interface_declaration", Kind: "interface", NameField: "name"},
			{Type: "type_alias_declaration", Kind: "type", NameField: "name"},
		}...),
	})
}
