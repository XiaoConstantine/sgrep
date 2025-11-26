package chunk

import (
	"unsafe"

	tree_sitter_c "github.com/tree-sitter/tree-sitter-c/bindings/go"
	tree_sitter_cpp "github.com/tree-sitter/tree-sitter-cpp/bindings/go"
)

func init() {
	cNodeTypes := []NodeTypeConfig{
		{Type: "function_definition", Kind: "function", NameField: "declarator"},
		{Type: "struct_specifier", Kind: "struct", NameField: "name"},
		{Type: "enum_specifier", Kind: "enum", NameField: "name"},
	}

	RegisterLanguage(&LanguageConfig{
		Name:       "c",
		Extensions: []string{".c", ".h"},
		Language:   func() unsafe.Pointer { return tree_sitter_c.Language() },
		NodeTypes:  cNodeTypes,
	})

	RegisterLanguage(&LanguageConfig{
		Name:       "cpp",
		Extensions: []string{".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"},
		Language:   func() unsafe.Pointer { return tree_sitter_cpp.Language() },
		NodeTypes: append(cNodeTypes, []NodeTypeConfig{
			{Type: "class_specifier", Kind: "class", NameField: "name"},
			{Type: "namespace_definition", Kind: "namespace", NameField: "name"},
			{Type: "template_declaration", Kind: "template", NameField: ""},
		}...),
	})
}
