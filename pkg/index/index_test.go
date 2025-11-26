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
