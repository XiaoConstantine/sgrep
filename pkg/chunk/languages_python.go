package chunk

import (
	"unsafe"

	tree_sitter_python "github.com/tree-sitter/tree-sitter-python/bindings/go"
)

func init() {
	RegisterLanguage(&LanguageConfig{
		Name:       "python",
		Extensions: []string{".py", ".pyw", ".pyi"},
		Language:   func() unsafe.Pointer { return tree_sitter_python.Language() },
		NodeTypes: []NodeTypeConfig{
			{Type: "function_definition", Kind: "function", NameField: "name", DocstringField: "body", DocstringType: "expression_statement"},
			{Type: "class_definition", Kind: "class", NameField: "name", DocstringField: "body", DocstringType: "expression_statement"},
			{Type: "decorated_definition", Kind: "decorated", NameField: ""},
		},
	})
}
