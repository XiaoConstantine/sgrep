package chunk

import (
	"unsafe"

	tree_sitter_java "github.com/tree-sitter/tree-sitter-java/bindings/go"
)

func init() {
	RegisterLanguage(&LanguageConfig{
		Name:       "java",
		Extensions: []string{".java"},
		Language:   func(string) unsafe.Pointer { return tree_sitter_java.Language() },
		NodeTypes: []NodeTypeConfig{
			{Type: "method_declaration", Kind: "method", NameField: "name", LeadingComment: true},
			{Type: "class_declaration", Kind: "class", NameField: "name", LeadingComment: true},
			{Type: "interface_declaration", Kind: "interface", NameField: "name", LeadingComment: true},
			{Type: "constructor_declaration", Kind: "constructor", NameField: "name", LeadingComment: true},
			{Type: "enum_declaration", Kind: "enum", NameField: "name", LeadingComment: true},
			{Type: "record_declaration", Kind: "record", NameField: "name", LeadingComment: true},
			{Type: "annotation_type_declaration", Kind: "annotation", NameField: "name", LeadingComment: true},
		},
	})
}
