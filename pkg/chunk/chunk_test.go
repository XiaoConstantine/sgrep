package chunk

import (
	"os"
	"strings"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.MaxTokens != defaultMaxTokens {
		t.Errorf("expected MaxTokens %d, got %d", defaultMaxTokens, cfg.MaxTokens)
	}
	if cfg.ContextLines != defaultContextLines {
		t.Errorf("expected ContextLines %d, got %d", defaultContextLines, cfg.ContextLines)
	}
	if cfg.Overlap != defaultOverlap {
		t.Errorf("expected Overlap %d, got %d", defaultOverlap, cfg.Overlap)
	}
}

func TestDefaultConfig_CustomMaxTokens(t *testing.T) {
	t.Setenv("SGREP_MAX_TOKENS", "500")
	defer func() { _ = os.Unsetenv("SGREP_MAX_TOKENS") }()

	cfg := DefaultConfig()
	if cfg.MaxTokens != 500 {
		t.Errorf("expected MaxTokens 500, got %d", cfg.MaxTokens)
	}
}

func TestDefaultConfig_InvalidEnv(t *testing.T) {
	t.Setenv("SGREP_MAX_TOKENS", "invalid")
	defer func() { _ = os.Unsetenv("SGREP_MAX_TOKENS") }()

	cfg := DefaultConfig()
	if cfg.MaxTokens != defaultMaxTokens {
		t.Errorf("expected default MaxTokens, got %d", cfg.MaxTokens)
	}
}

func TestChunkFile_Go(t *testing.T) {
	content := `package main

import "fmt"

// HelloWorld prints hello world and does various things to make this function
// substantial enough to be chunked by the AST parser which has minimum token thresholds.
func HelloWorld() {
	fmt.Println("hello world")
	fmt.Println("more content here to ensure we have enough tokens")
	fmt.Println("additional lines to make the function larger")
	for i := 0; i < 10; i++ {
		fmt.Printf("iteration %d\n", i)
	}
}

// Add adds two numbers together and returns the result.
// This function demonstrates basic arithmetic operations in Go.
func Add(a, b int) int {
	result := a + b
	fmt.Printf("Adding %d + %d = %d\n", a, b, result)
	return result
}
`

	chunks, err := ChunkFile("/test/main.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least 1 chunk")
	}

	foundHello := false
	foundAdd := false
	for _, c := range chunks {
		if strings.Contains(c.Description, "HelloWorld") {
			foundHello = true
		}
		if strings.Contains(c.Description, "Add") {
			foundAdd = true
		}
	}

	if len(chunks) >= 2 && !foundHello {
		t.Error("expected to find HelloWorld function when we have 2+ chunks")
	}
	if len(chunks) >= 2 && !foundAdd {
		t.Error("expected to find Add function when we have 2+ chunks")
	}
}

func TestChunkFile_GoWithReceiver(t *testing.T) {
	content := `package main

import "fmt"

type Server struct {
	port int
	host string
	name string
}

// Start starts the server and begins listening for connections.
// It performs various initialization steps and logging.
func (s *Server) Start() error {
	fmt.Printf("Starting server %s on %s:%d\n", s.name, s.host, s.port)
	fmt.Println("Initializing server components...")
	fmt.Println("Loading configuration...")
	fmt.Println("Server started successfully")
	return nil
}
`

	chunks, err := ChunkFile("/test/server.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
		return
	}

	foundServerRelated := false
	for _, c := range chunks {
		if strings.Contains(c.Description, "Start") ||
			strings.Contains(c.Description, "Server") ||
			strings.Contains(c.Content, "Server") {
			foundServerRelated = true
		}
	}

	if !foundServerRelated {
		t.Error("expected to find Server-related content")
	}
}

func TestChunkFile_GoTypes(t *testing.T) {
	content := `package main

// Config holds configuration settings for the application.
// It contains multiple fields for various settings.
type Config struct {
	Host     string
	Port     int
	Debug    bool
	LogLevel string
	Timeout  int
	Workers  int
}

// Handler is an interface for handlers that process requests.
// Implementations must provide Handle and Close methods.
type Handler interface {
	Handle(data []byte) error
	Close() error
	Name() string
}
`

	chunks, err := ChunkFile("/test/types.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
		return
	}

	foundConfig := false
	foundHandler := false
	for _, c := range chunks {
		if strings.Contains(c.Description, "Config") || strings.Contains(c.Content, "Config") {
			foundConfig = true
		}
		if strings.Contains(c.Description, "Handler") || strings.Contains(c.Content, "Handler") {
			foundHandler = true
		}
	}

	if !foundConfig {
		t.Error("expected to find Config-related content")
	}
	if !foundHandler {
		t.Error("expected to find Handler-related content")
	}
}

func TestChunkFile_GoInvalidSyntax(t *testing.T) {
	content := `package main

func broken() {
	// missing closing brace
`

	chunks, err := ChunkFile("/test/broken.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("should fall back to size-based chunking")
	}
}

func TestChunkFile_NonGo(t *testing.T) {
	content := strings.Repeat("some content line\n", 100)

	chunks, err := ChunkFile("/test/file.txt", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkFile_TypeScript(t *testing.T) {
	content := `
export function hello(): void {
	console.log("hello");
}

export class Service {
	run(): void {}
}
`

	chunks, err := ChunkFile("/test/file.ts", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkFile_Python(t *testing.T) {
	content := `
def hello():
    print("hello")

class Service:
    def run(self):
        pass
`

	chunks, err := ChunkFile("/test/file.py", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkBySize(t *testing.T) {
	lines := make([]string, 50)
	for i := range lines {
		lines[i] = "This is line " + string(rune('0'+i%10))
	}
	content := strings.Join(lines, "\n")

	cfg := &Config{
		MaxTokens:    20,
		ContextLines: 5,
		Overlap:      2,
	}

	chunks, err := chunkBySize("/test/file.txt", content, cfg)
	if err != nil {
		t.Fatalf("chunkBySize failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}

	for _, c := range chunks {
		if c.FilePath != "/test/file.txt" {
			t.Errorf("expected filepath /test/file.txt, got %s", c.FilePath)
		}
		if c.StartLine < 1 {
			t.Errorf("invalid start line: %d", c.StartLine)
		}
		if c.EndLine < c.StartLine {
			t.Errorf("end line %d < start line %d", c.EndLine, c.StartLine)
		}
	}
}

func TestChunkBySize_Empty(t *testing.T) {
	cfg := DefaultConfig()
	chunks, err := chunkBySize("/test/empty.txt", "", cfg)
	if err != nil {
		t.Fatalf("chunkBySize failed: %v", err)
	}
	if len(chunks) > 1 {
		t.Errorf("expected at most 1 chunk for empty content, got %d", len(chunks))
	}
}

func TestSplitOversized(t *testing.T) {
	lines := make([]string, 100)
	for i := range lines {
		lines[i] = "Long content line " + string(rune('0'+i%10))
	}

	chunk := Chunk{
		Content:     strings.Join(lines, "\n"),
		StartLine:   1,
		EndLine:     100,
		FilePath:    "/test/file.go",
		Description: "func BigFunction",
	}

	cfg := &Config{
		MaxTokens: 50,
		Overlap:   0,
	}

	chunks := splitOversized(chunk, cfg)

	if len(chunks) <= 1 {
		t.Error("expected multiple chunks after splitting")
	}

	for i, c := range chunks {
		if !strings.Contains(c.Description, "func BigFunction") {
			t.Errorf("chunk %d missing original description", i)
		}
		if i > 0 && !strings.Contains(c.Description, "part") {
			t.Logf("chunk %d description: %s", i, c.Description)
		}
	}
}

func TestBuildFuncDescription(t *testing.T) {
	content := `package main

// Add adds two numbers.
func Add(a, b int) int {
	return a + b
}
`

	chunks, _ := ChunkFile("/test/math.go", content, nil)

	found := false
	for _, c := range chunks {
		if strings.Contains(c.Description, "Add") {
			found = true
			if !strings.Contains(c.Description, "package main") {
				t.Error("description should contain package name")
			}
			if !strings.Contains(c.Description, "math.go") {
				t.Error("description should contain file name")
			}
		}
	}

	if !found {
		t.Error("expected to find Add function")
	}
}

func TestBuildTypeDescription(t *testing.T) {
	content := `package model

// User represents a user in the system with multiple fields
// for storing user information and preferences.
type User struct {
	Name      string
	Email     string
	Age       int
	Active    bool
	Roles     []string
	CreatedAt int64
}
`

	chunks, _ := ChunkFile("/test/user.go", content, nil)

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
		return
	}

	found := false
	for _, c := range chunks {
		if strings.Contains(c.Description, "User") || strings.Contains(c.Content, "User") {
			found = true
		}
	}

	if !found {
		t.Error("expected to find User content")
	}
}

func TestBuildSizeDescription(t *testing.T) {
	desc := buildSizeDescription("/path/to/file.txt", 10, 25)

	if !strings.Contains(desc, "file.txt") {
		t.Error("description should contain file name")
	}
	if !strings.Contains(desc, "10") || !strings.Contains(desc, "25") {
		t.Error("description should contain line numbers")
	}
}

func TestFormatType(t *testing.T) {
	content := `package main

func (s *Server) Handle() {}
func (c Client) Process() {}
`

	chunks, _ := ChunkFile("/test/types.go", content, nil)

	if len(chunks) == 0 {
		t.Error("expected chunks")
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		text     string
		minToken int
		maxToken int
	}{
		{"", 0, 0},
		{"hello", 1, 3},
		{"hello world", 2, 5},
		{"one two three four five", 5, 10},
	}

	for _, tt := range tests {
		tokens := estimateTokens(tt.text)
		if tokens < tt.minToken || tokens > tt.maxToken {
			t.Errorf("estimateTokens(%q) = %d, expected %d-%d",
				tt.text, tokens, tt.minToken, tt.maxToken)
		}
	}
}

func TestMax(t *testing.T) {
	if max(5, 10) != 10 {
		t.Error("max(5, 10) should be 10")
	}
	if max(10, 5) != 10 {
		t.Error("max(10, 5) should be 10")
	}
	if max(5, 5) != 5 {
		t.Error("max(5, 5) should be 5")
	}
}

func TestChunk_Fields(t *testing.T) {
	c := Chunk{
		Content:     "func main() {}",
		StartLine:   1,
		EndLine:     3,
		FilePath:    "/test/main.go",
		Description: "Go function main",
	}

	if c.Content != "func main() {}" {
		t.Error("unexpected Content")
	}
	if c.StartLine != 1 {
		t.Error("unexpected StartLine")
	}
	if c.EndLine != 3 {
		t.Error("unexpected EndLine")
	}
	if c.FilePath != "/test/main.go" {
		t.Error("unexpected FilePath")
	}
	if c.Description != "Go function main" {
		t.Error("unexpected Description")
	}
}

func TestChunkFile_LargeFile(t *testing.T) {
	lines := make([]string, 1000)
	for i := range lines {
		lines[i] = "// Line " + string(rune('0'+i%10))
	}
	content := "package main\n\nfunc main() {\n" + strings.Join(lines, "\n") + "\n}"

	chunks, err := ChunkFile("/test/large.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected chunks for large file")
	}
}

func TestChunkFile_NilConfig(t *testing.T) {
	content := "package main\n\nfunc main() {}"

	chunks, err := ChunkFile("/test/main.go", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed with nil config: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

// Benchmarks

func BenchmarkChunkFile_Go(b *testing.B) {
	content := `package main

import "fmt"

func HelloWorld() {
	fmt.Println("hello")
}

func Add(a, b int) int {
	return a + b
}

type Server struct {
	port int
}

func (s *Server) Start() error {
	return nil
}
`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ChunkFile("/test/main.go", content, nil)
	}
}

func BenchmarkChunkBySize(b *testing.B) {
	lines := make([]string, 500)
	for i := range lines {
		lines[i] = "This is line content for benchmarking."
	}
	content := strings.Join(lines, "\n")
	cfg := DefaultConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = chunkBySize("/test/file.txt", content, cfg)
	}
}

func BenchmarkEstimateTokens(b *testing.B) {
	text := "This is a sample text for token estimation benchmarking."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		estimateTokens(text)
	}
}

func TestSplitLongLine(t *testing.T) {
	tests := []struct {
		name      string
		line      string
		maxTokens int
		minParts  int
	}{
		{
			name:      "empty line",
			line:      "",
			maxTokens: 10,
			minParts:  1,
		},
		{
			name:      "short line",
			line:      "hello world",
			maxTokens: 100,
			minParts:  1,
		},
		{
			name:      "long line needs splitting",
			line:      "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10",
			maxTokens: 5,
			minParts:  2,
		},
		{
			name:      "single very long word",
			line:      "superlongwordthatexceedsthelimit",
			maxTokens: 2,
			minParts:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitLongLine(tt.line, tt.maxTokens)
			if len(result) < tt.minParts {
				t.Errorf("expected at least %d parts, got %d", tt.minParts, len(result))
			}
		})
	}
}

func TestChunkBySize_LongLines(t *testing.T) {
	// Create content with a very long line
	longLine := strings.Repeat("word ", 500)
	content := "short line\n" + longLine + "\nanother short line"

	cfg := &Config{
		MaxTokens:    50,
		ContextLines: 5,
		Overlap:      2,
	}

	chunks, err := chunkBySize("/test/file.txt", content, cfg)
	if err != nil {
		t.Fatalf("chunkBySize failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkFile_Rust(t *testing.T) {
	content := `
fn main() {
    println!("Hello, world!");
}

struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}
`

	chunks, err := ChunkFile("/test/main.rs", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkFile_JavaScript(t *testing.T) {
	content := `
function hello() {
    console.log("hello");
}

class Service {
    constructor() {
        this.name = "service";
    }

    run() {
        return true;
    }
}

const arrow = () => {
    return 42;
};
`

	chunks, err := ChunkFile("/test/app.js", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestChunkFile_EmptyContent(t *testing.T) {
	chunks, err := ChunkFile("/test/empty.go", "", nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	// Empty content should produce no or minimal chunks
	if len(chunks) > 1 {
		t.Errorf("expected at most 1 chunk for empty content, got %d", len(chunks))
	}
}

func TestSplitOversized_WithVeryLongLine(t *testing.T) {
	// Create a chunk with a very long line
	longLine := strings.Repeat("longword ", 200)
	chunk := Chunk{
		Content:     "short line\n" + longLine + "\nshort line",
		StartLine:   1,
		EndLine:     3,
		FilePath:    "/test/file.go",
		Description: "test chunk",
	}

	cfg := &Config{
		MaxTokens: 30,
		Overlap:   0,
	}

	chunks := splitOversized(chunk, cfg)

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestConfig_Fields(t *testing.T) {
	cfg := &Config{
		MaxTokens:    500,
		ContextLines: 5,
		Overlap:      2,
	}

	if cfg.MaxTokens != 500 {
		t.Error("unexpected MaxTokens")
	}
	if cfg.ContextLines != 5 {
		t.Error("unexpected ContextLines")
	}
	if cfg.Overlap != 2 {
		t.Error("unexpected Overlap")
	}
}

func TestChunkBySize_SmallMaxTokens(t *testing.T) {
	// Create multi-line content to ensure chunking
	lines := make([]string, 50)
	for i := range lines {
		lines[i] = "word word word"
	}
	content := strings.Join(lines, "\n")

	cfg := &Config{
		MaxTokens:    20, // Small to force multiple chunks
		ContextLines: 2,
		Overlap:      1,
	}

	chunks, err := chunkBySize("/test/file.txt", content, cfg)
	if err != nil {
		t.Fatalf("chunkBySize failed: %v", err)
	}

	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks with small MaxTokens, got %d", len(chunks))
	}
}
