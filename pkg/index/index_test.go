package index

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	searchpkg "github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/util"
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
	ir.ensureRulesLoaded("/nonexistent")
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

func TestDecomposeDocumentForColBERT_AdaptiveUsesRawSegments(t *testing.T) {
	content := makeSegmentedContent(16)

	raw := searchpkg.DecomposeDocumentRaw(content)
	adaptive := decomposeDocumentForColBERT(content, true)
	legacy := decomposeDocumentForColBERT(content, false)
	budget := searchpkg.AdaptiveSegmentBudget(content)

	if len(raw) <= budget {
		t.Fatalf("test content did not exceed adaptive budget: raw=%d budget=%d", len(raw), budget)
	}
	if len(adaptive) != len(raw) {
		t.Fatalf("adaptive decomposition should keep raw segments before pooling: got %d want %d", len(adaptive), len(raw))
	}
	if len(legacy) >= len(adaptive) {
		t.Fatalf("legacy decomposition should stay capped below raw adaptive split: legacy=%d adaptive=%d", len(legacy), len(adaptive))
	}
}

func TestBuildStoredColBERTSegments_AdaptivePoolsToBudget(t *testing.T) {
	segmentTexts := make([]string, 16)
	embeddings := make([][]float32, len(segmentTexts))
	budget := searchpkg.AdaptiveSegmentBudgetFromRawCount(len(segmentTexts))

	rng := rand.New(rand.NewSource(42))
	for i := range segmentTexts {
		segmentTexts[i] = fmt.Sprintf("segment %d", i)
		embeddings[i] = randomNormalizedEmbedding(rng, 768)
	}

	segments := buildStoredColBERTSegments(segmentTexts, embeddings, true)

	if len(segments) > budget {
		t.Fatalf("adaptive pooling exceeded budget: got %d want <= %d", len(segments), budget)
	}
	if len(segments) >= len(segmentTexts) {
		t.Fatalf("adaptive pooling did not reduce segment count: got %d from %d", len(segments), len(segmentTexts))
	}
	for i, seg := range segments {
		if seg.SegmentIdx != i {
			t.Fatalf("segment %d has idx %d", i, seg.SegmentIdx)
		}
	}
}

func TestBuildStoredColBERTSegments_AdaptiveKeepsLegacySizedChunks(t *testing.T) {
	segmentTexts := make([]string, 6)
	embeddings := make([][]float32, len(segmentTexts))

	rng := rand.New(rand.NewSource(99))
	for i := range segmentTexts {
		segmentTexts[i] = fmt.Sprintf("segment %d", i)
		embeddings[i] = randomNormalizedEmbedding(rng, 768)
	}

	segments := buildStoredColBERTSegments(segmentTexts, embeddings, true)
	if len(segments) != len(segmentTexts) {
		t.Fatalf("legacy-sized chunk should remain unchanged: got %d want %d", len(segments), len(segmentTexts))
	}
}

func TestBuildStoredColBERTSegments_AdaptiveMergesSimilarSegments(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	baseA := randomNormalizedEmbedding(rng, 768)
	baseB := randomNormalizedEmbedding(rng, 768)
	baseC := randomNormalizedEmbedding(rng, 768)
	baseD := randomNormalizedEmbedding(rng, 768)

	segmentTexts := []string{"a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2", "d0", "d1", "d2"}
	embeddings := [][]float32{
		perturbEmbedding(baseA, 0.002),
		perturbEmbedding(baseA, 0.003),
		perturbEmbedding(baseA, 0.004),
		perturbEmbedding(baseB, 0.002),
		perturbEmbedding(baseB, 0.003),
		perturbEmbedding(baseB, 0.004),
		perturbEmbedding(baseC, 0.002),
		perturbEmbedding(baseC, 0.003),
		perturbEmbedding(baseC, 0.004),
		perturbEmbedding(baseD, 0.002),
		perturbEmbedding(baseD, 0.003),
		perturbEmbedding(baseD, 0.004),
	}

	segments := buildStoredColBERTSegments(segmentTexts, embeddings, true)
	if len(segments) >= len(segmentTexts) {
		t.Fatalf("adaptive merge did not collapse similar segments: got %d from %d", len(segments), len(segmentTexts))
	}
	if len(segments) > searchpkg.AdaptiveSegmentBudgetFromRawCount(len(segmentTexts)) {
		t.Fatalf("expected merged result to respect adaptive budget, got %d", len(segments))
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

func makeSegmentedContent(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "func segment%d() {\n", i)
		fmt.Fprintf(&b, "    value%d := %d\n", i, i)
		fmt.Fprintf(&b, "    return value%d\n", i)
		b.WriteString("}\n\n")
	}
	return b.String()
}

func randomNormalizedEmbedding(rng *rand.Rand, dims int) []float32 {
	emb := make([]float32, dims)
	for i := range emb {
		emb[i] = rng.Float32()*2 - 1
	}
	return util.NormalizeVector(emb)
}

func perturbEmbedding(base []float32, noise float32) []float32 {
	emb := make([]float32, len(base))
	copy(emb, base)
	for i := range emb {
		delta := noise
		if i%2 == 0 {
			delta = -noise
		}
		emb[i] += delta
	}
	return util.NormalizeVector(emb)
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

func TestIgnoreRules_NestedGitignore(t *testing.T) {
	dir := t.TempDir()
	genDir := filepath.Join(dir, "pkg", "generated")
	if err := os.MkdirAll(genDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pkg", ".gitignore"), []byte("generated/\n"), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnore(filepath.Join(genDir, "client.go")) {
		t.Fatal("nested .gitignore directory rule should ignore descendants")
	}
	if ir.ShouldIgnore(filepath.Join(dir, "pkg", "service.go")) {
		t.Fatal("nested .gitignore should not ignore siblings outside the rule")
	}
}

func TestIgnoreRules_Negation(t *testing.T) {
	dir := t.TempDir()
	content := "*.go\n!keep.go\n"
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnore(filepath.Join(dir, "drop.go")) {
		t.Fatal("expected generic ignore pattern to match")
	}
	if ir.ShouldIgnore(filepath.Join(dir, "keep.go")) {
		t.Fatal("negated rule should re-include keep.go")
	}
}

func TestIgnoreRules_AnchoredPattern(t *testing.T) {
	dir := t.TempDir()
	content := "/generated/\n"
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	ir := NewIgnoreRules(dir)

	if !ir.ShouldIgnorePath(filepath.Join(dir, "generated"), true) {
		t.Fatal("anchored root directory should be ignored")
	}
	if ir.ShouldIgnorePath(filepath.Join(dir, "pkg", "generated"), true) {
		t.Fatal("anchored pattern should not match nested directory")
	}
}
