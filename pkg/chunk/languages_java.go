package chunk

import (
	"unsafe"

	tree_sitter_java "github.com/tree-sitter/tree-sitter-java/bindings/go"
)

func init() {
	RegisterLanguage(&LanguageConfig{
		Name:       "java",
		Extensions: []string{".java"},
		Language:   func() unsafe.Pointer { return tree_sitter_java.Language() },
		NodeTypes: []NodeTypeConfig{
			{Type: "method_declaration", Kind: "method", NameField: "name"},
			{Type: "class_declaration", Kind: "class", NameField: "name"},
			{Type: "interface_declaration", Kind: "interface", NameField: "name"},
			{Type: "constructor_declaration", Kind: "constructor", NameField: "name"},
			{Type: "enum_declaration", Kind: "enum", NameField: "name"},
		},
	})
}
