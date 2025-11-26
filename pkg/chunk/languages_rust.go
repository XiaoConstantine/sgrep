package chunk

import (
	"unsafe"

	tree_sitter_rust "github.com/tree-sitter/tree-sitter-rust/bindings/go"
)

func init() {
	RegisterLanguage(&LanguageConfig{
		Name:       "rust",
		Extensions: []string{".rs"},
		Language:   func() unsafe.Pointer { return tree_sitter_rust.Language() },
		NodeTypes: []NodeTypeConfig{
			{Type: "function_item", Kind: "function", NameField: "name", LeadingComment: true},
			{Type: "impl_item", Kind: "impl", NameField: "type", LeadingComment: true},
			{Type: "struct_item", Kind: "struct", NameField: "name", LeadingComment: true},
			{Type: "enum_item", Kind: "enum", NameField: "name", LeadingComment: true},
			{Type: "trait_item", Kind: "trait", NameField: "name", LeadingComment: true},
		},
	})
}
