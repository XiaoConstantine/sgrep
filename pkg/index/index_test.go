package index

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetSgrepHome(t *testing.T) {
	t.Run("from_env", func(t *testing.T) {
		t.Setenv("SGREP_HOME", "/custom/path")
		home, err := getSgrepHome()
		if err != nil {
			t.Fatal(err)
		}
		if home != "/custom/path" {
			t.Errorf("got %s, want /custom/path", home)
		}
	})

	t.Run("default", func(t *testing.T) {
		t.Setenv("SGREP_HOME", "")
		home, err := getSgrepHome()
		if err != nil {
			t.Fatal(err)
		}
		homeDir, _ := os.UserHomeDir()
		if home != filepath.Join(homeDir, ".sgrep") {
			t.Errorf("got %s", home)
		}
	})
}

func TestHashPath(t *testing.T) {
	h1 := hashPath("/path/one")
	h2 := hashPath("/path/two")
	if len(h1) != 12 || h1 == h2 {
		t.Errorf("hash issue: %s vs %s", h1, h2)
	}
	if hashPath("/path/one") != h1 {
		t.Error("not deterministic")
	}
}

func TestWriteRepoMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := writeRepoMetadata(dir, "/test/repo"); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if len(data) == 0 {
		t.Error("empty metadata")
	}
}

func TestIsCodeFile(t *testing.T) {
	cases := map[string]bool{
		"main.go": true, "app.ts": true, "x.py": true, "x.rs": true,
		"x.java": true, "x.c": true, "x.cpp": true, "x.rb": true,
		"x.md": true, "x.json": true, "x.yaml": true, "x.toml": true,
		"x.png": false, "x.exe": false, "noext": false,
	}
	for path, want := range cases {
		if got := isCodeFile(path); got != want {
			t.Errorf("isCodeFile(%q)=%v want %v", path, got, want)
		}
	}
}

func TestIsKnownIgnoreDir(t *testing.T) {
	for _, d := range []string{"node_modules", "vendor", ".git", "dist", "build"} {
		if !isKnownIgnoreDir(d) {
			t.Errorf("%s should be ignored", d)
		}
	}
	if isKnownIgnoreDir("src") {
		t.Error("src should not be ignored")
	}
}

func TestIgnoreRules(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("*.log\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, ".sgrepignore"), []byte("custom/\n"), 0644)

	ir := NewIgnoreRules(dir)

	tests := []struct {
		path string
		want bool
	}{
		{dir, false},
		{filepath.Join(dir, "node_modules"), true},
		{filepath.Join(dir, ".git"), true},
		{filepath.Join(dir, "src"), false},
		{filepath.Join(dir, "app.log"), true},
		{filepath.Join(dir, "app.min.js"), true},
	}
	for _, tt := range tests {
		if got := ir.ShouldIgnore(tt.path); got != tt.want {
			t.Errorf("ShouldIgnore(%q)=%v want %v", tt.path, got, tt.want)
		}
	}
}

func TestIgnoreRules_LoadMissing(t *testing.T) {
	ir := &IgnoreRules{rootPath: "/nonexistent"}
	ir.loadIgnoreFile("/nonexistent/.gitignore")
}

func TestIndexer_Fields(t *testing.T) {
	idx := &Indexer{rootPath: "/test", processed: 5, errors: 2}
	if idx.rootPath != "/test" || idx.processed != 5 || idx.errors != 2 {
		t.Error("field issue")
	}
}

func TestDefaultIndexConfig(t *testing.T) {
	cfg := DefaultIndexConfig()
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if cfg.Workers < 4 {
		t.Errorf("workers should be at least 4, got %d", cfg.Workers)
	}
	if cfg.Workers > 32 {
		t.Errorf("workers should be capped at 32, got %d", cfg.Workers)
	}
	if cfg.EmbedConcurrency < 4 {
		t.Errorf("EmbedConcurrency should be at least 4, got %d", cfg.EmbedConcurrency)
	}
	if cfg.EmbedConcurrency > 16 {
		t.Errorf("EmbedConcurrency should be capped at 16, got %d", cfg.EmbedConcurrency)
	}
}

func TestIsTestFile(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		// Go test files
		{"main_test.go", true},
		{"pkg/foo_test.go", true},
		{"main.go", false},

		// JS/TS test files
		{"app.test.ts", true},
		{"app.test.tsx", true},
		{"app.test.js", true},
		{"app.test.jsx", true},
		{"app.spec.ts", true},
		{"app.spec.tsx", true},
		{"app.spec.js", true},
		{"app.spec.jsx", true},
		{"app.ts", false},
		{"app.js", false},

		// Python test files
		{"test_main.py", true},
		{"main_test.py", true},
		{"main.py", false},

		// Ruby test files
		{"main_spec.rb", true},
		{"main.rb", false},

		// Rust test files
		{"main_test.rs", true},
		{"main.rs", false},

		// Java test files
		{"MainTest.java", true},
		{"MainTests.java", true},
		{"Main.java", false},

		// Files in test directories (need proper path structure)
		{filepath.Join("src", "tests", "main.go"), true},
		{filepath.Join("src", "test", "main.py"), true},
		{filepath.Join("src", "__tests__", "app.js"), true},
		{filepath.Join("src", "spec", "helper.rb"), true},
		{filepath.Join("src", "specs", "main.rb"), true},
		{filepath.Join("src", "_tests", "foo.go"), true},

		// Non-test files
		{"src/main.go", false},
		{"lib/util.py", false},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got := isTestFile(tt.path)
			if got != tt.want {
				t.Errorf("isTestFile(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestTruncateAtBoundary(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		maxChars int
		want     string
	}{
		{
			name:     "no truncation needed",
			text:     "hello world",
			maxChars: 20,
			want:     "hello world",
		},
		{
			name:     "truncate at line boundary",
			text:     "line one\nline two\nline three",
			maxChars: 20,
			want:     "line one\nline two",
		},
		{
			name:     "truncate at word boundary",
			text:     "hello beautiful world today",
			maxChars: 15,
			want:     "hello beautiful",
		},
		{
			name:     "hard truncate when no good boundary",
			text:     "abcdefghijklmnop",
			maxChars: 10,
			want:     "abcdefghij",
		},
		{
			name:     "empty string",
			text:     "",
			maxChars: 10,
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := truncateAtBoundary(tt.text, tt.maxChars)
			if len(got) > tt.maxChars {
				t.Errorf("truncateAtBoundary() returned string longer than maxChars: len=%d, max=%d", len(got), tt.maxChars)
			}
		})
	}
}

func TestIgnoreRules_Patterns(t *testing.T) {
	dir := t.TempDir()

	// Create .gitignore with various patterns
	gitignore := `# Comment
*.log
build/
*.min.js
`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(gitignore), 0644)

	ir := NewIgnoreRules(dir)

	tests := []struct {
		path string
		want bool
	}{
		// Default ignores
		{filepath.Join(dir, "node_modules"), true},
		{filepath.Join(dir, "vendor"), true},
		{filepath.Join(dir, "__pycache__"), true},
		{filepath.Join(dir, ".idea"), true},
		{filepath.Join(dir, ".vscode"), true},
		{filepath.Join(dir, "dist"), true},
		{filepath.Join(dir, "build"), true},
		{filepath.Join(dir, ".git"), true},
		{filepath.Join(dir, ".sgrep"), true},

		// From .gitignore
		{filepath.Join(dir, "app.log"), true},
		{filepath.Join(dir, "bundle.min.js"), true},

		// Should not ignore
		{filepath.Join(dir, "src"), false},
		{filepath.Join(dir, "main.go"), false},
		{filepath.Join(dir, "app.js"), false},
	}

	for _, tt := range tests {
		t.Run(filepath.Base(tt.path), func(t *testing.T) {
			got := ir.ShouldIgnore(tt.path)
			if got != tt.want {
				t.Errorf("ShouldIgnore(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestIgnoreRules_CommentLines(t *testing.T) {
	dir := t.TempDir()

	// .gitignore with comments
	content := `# This is a comment
*.log
# Another comment
`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644)

	ir := NewIgnoreRules(dir)

	// Comments should not be treated as patterns
	if ir.ShouldIgnore(filepath.Join(dir, "# This is a comment")) {
		t.Error("comment line should not be used as pattern")
	}
}

func TestIgnoreRules_EmptyLines(t *testing.T) {
	dir := t.TempDir()

	content := `*.log

*.tmp

`
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644)

	ir := NewIgnoreRules(dir)

	// Should ignore .log and .tmp files
	if !ir.ShouldIgnore(filepath.Join(dir, "app.log")) {
		t.Error("should ignore .log files")
	}
	if !ir.ShouldIgnore(filepath.Join(dir, "temp.tmp")) {
		t.Error("should ignore .tmp files")
	}
}
