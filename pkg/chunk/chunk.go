package chunk

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
)

var defaultMaxTokens = modelcfg.DocumentTokenBudget()

const (
	defaultContextLines = 10
	defaultOverlap      = 3
)

// Chunk represents a code chunk for indexing.
type Chunk struct {
	Content     string
	StartLine   int
	EndLine     int
	FilePath    string
	Description string // AST-derived description for better embeddings
}

// Config holds chunking configuration.
type Config struct {
	MaxTokens    int
	ContextLines int
	Overlap      int
}

// DefaultConfig returns the default chunking config.
func DefaultConfig() *Config {
	maxTokens := defaultMaxTokens
	if v := os.Getenv("SGREP_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}

	return &Config{
		MaxTokens:    maxTokens,
		ContextLines: defaultContextLines,
		Overlap:      defaultOverlap,
	}
}

// ChunkFile splits a file into chunks based on its type.
func ChunkFile(path string, content string, cfg *Config) ([]Chunk, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	ext := strings.ToLower(filepath.Ext(path))

	// Special case: Keep using Go's native AST (more accurate)
	if ext == ".go" {
		return chunkGo(path, content, cfg)
	}

	// Try tree-sitter for registered languages
	if langCfg := GetLanguageByPath(path); langCfg != nil {
		return chunkTreeSitter(path, content, cfg, langCfg)
	}

	// Fallback to size-based chunking
	return chunkBySize(path, content, cfg)
}

// chunkGo uses Go AST to split at function/type boundaries.
func chunkGo(path string, content string, cfg *Config) ([]Chunk, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, content, parser.ParseComments)
	if err != nil {
		// Fallback to size-based if parse fails
		return chunkBySize(path, content, cfg)
	}

	var chunks []Chunk
	baseName := filepath.Base(path)
	pkgName := file.Name.Name

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			startPos := fset.Position(d.Pos())
			endPos := fset.Position(d.End())

			// Include doc comments
			var chunkContent string
			if d.Doc != nil {
				docPos := fset.Position(d.Doc.Pos())
				chunkContent = content[d.Doc.Pos()-1 : d.End()-1]
				startPos = docPos
			} else {
				chunkContent = content[d.Pos()-1 : d.End()-1]
			}

			// Skip if too small
			if estimateTokens(chunkContent) < 20 {
				continue
			}

			chunk := Chunk{
				Content:     chunkContent,
				StartLine:   startPos.Line,
				EndLine:     endPos.Line,
				FilePath:    path,
				Description: buildFuncDescription(baseName, pkgName, d),
			}
			chunks = append(chunks, chunk)

		case *ast.GenDecl:
			if d.Tok == token.TYPE {
				for _, spec := range d.Specs {
					if ts, ok := spec.(*ast.TypeSpec); ok {
						startPos := fset.Position(ts.Pos())
						endPos := fset.Position(ts.End())

						var chunkContent string
						if d.Doc != nil {
							docPos := fset.Position(d.Doc.Pos())
							chunkContent = content[d.Doc.Pos()-1 : ts.End()-1]
							startPos = docPos
						} else {
							chunkContent = content[ts.Pos()-1 : ts.End()-1]
						}

						if estimateTokens(chunkContent) < 20 {
							continue
						}

						chunk := Chunk{
							Content:     chunkContent,
							StartLine:   startPos.Line,
							EndLine:     endPos.Line,
							FilePath:    path,
							Description: buildTypeDescription(baseName, pkgName, ts),
						}
						chunks = append(chunks, chunk)
					}
				}
			}
		}
	}

	// If no AST chunks, fall back to size-based
	if len(chunks) == 0 {
		return chunkBySize(path, content, cfg)
	}

	// Split oversized chunks
	var finalChunks []Chunk
	for _, chunk := range chunks {
		if estimateTokens(chunk.Content) > cfg.MaxTokens {
			subChunks := SplitChunk(chunk, cfg)
			finalChunks = append(finalChunks, subChunks...)
		} else {
			finalChunks = append(finalChunks, chunk)
		}
	}

	return finalChunks, nil
}

// chunkBySize splits content into fixed-size chunks.
func chunkBySize(path string, content string, cfg *Config) ([]Chunk, error) {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 {
		return nil, nil
	}

	// Reserve tokens for description overhead
	descOverhead := 50 // Approximate tokens for "Code from file.go (lines X-Y)"
	effectiveMax := cfg.MaxTokens - descOverhead
	if effectiveMax < 100 {
		effectiveMax = 100
	}

	var chunks []Chunk
	var currentLines []string
	currentTokens := 0
	startLine := 1

	for i, line := range lines {
		lineTokens := estimateTokens(line)

		// Handle single lines that exceed the limit
		if lineTokens > effectiveMax {
			// Flush current chunk first
			if len(currentLines) > 0 {
				chunk := Chunk{
					Content:     strings.Join(currentLines, "\n"),
					StartLine:   startLine,
					EndLine:     i,
					FilePath:    path,
					Description: buildSizeDescription(path, startLine, i),
				}
				chunks = append(chunks, chunk)
				currentLines = nil
				currentTokens = 0
			}

			// Split the long line
			splitLines := splitLongLine(line, effectiveMax)
			for _, sl := range splitLines {
				chunk := Chunk{
					Content:     sl,
					StartLine:   i + 1,
					EndLine:     i + 1,
					FilePath:    path,
					Description: buildSizeDescription(path, i+1, i+1),
				}
				chunks = append(chunks, chunk)
			}
			startLine = i + 2
			continue
		}

		if currentTokens+lineTokens > effectiveMax && len(currentLines) > 0 {
			chunk := Chunk{
				Content:     strings.Join(currentLines, "\n"),
				StartLine:   startLine,
				EndLine:     i,
				FilePath:    path,
				Description: buildSizeDescription(path, startLine, i),
			}
			chunks = append(chunks, chunk)

			// Start new chunk with overlap
			overlapStart := max(0, len(currentLines)-cfg.Overlap)
			currentLines = currentLines[overlapStart:]
			currentTokens = estimateTokens(strings.Join(currentLines, "\n"))
			startLine = i - len(currentLines) + 1
		}

		currentLines = append(currentLines, line)
		currentTokens += lineTokens
	}

	// Final chunk
	if len(currentLines) > 0 {
		chunk := Chunk{
			Content:     strings.Join(currentLines, "\n"),
			StartLine:   startLine,
			EndLine:     len(lines),
			FilePath:    path,
			Description: buildSizeDescription(path, startLine, len(lines)),
		}
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// SplitChunk splits an existing chunk without extracting or discarding source.
// It keeps syntax groups together when they fit, with line/word splitting as a
// fallback. MaxTokens includes the description; an indivisible oversized word or
// description is preserved so the embedding validator can reject it explicitly.
func SplitChunk(chunk Chunk, cfg *Config) []Chunk {
	lines := strings.Split(chunk.Content, "\n")
	var chunks []Chunk
	var currentLines []string
	currentTokens := 0
	startLine := chunk.StartLine

	// Reserve tokens for description overhead (description + "\n\n" separator)
	descTokens := estimateTokens(chunk.Description) + 10 // +10 buffer for separator and part suffix
	effectiveMax := cfg.MaxTokens - descTokens
	if effectiveMax <= 0 {
		return []Chunk{chunk}
	}

	lineCosts := make([]int, len(lines))
	for i, line := range lines {
		// Include separators and per-line rounding: the word-based estimate rounds
		// down, so summing bare line estimates can undercount the joined content.
		lineCosts[i] = estimateTokens(line+"\n") + 1
	}
	groupCosts := treeSitterGroupTokens(chunk.FilePath, chunk.Content, lineCosts, effectiveMax)

	for i, line := range lines {
		lineTokens := lineCosts[i]

		// Handle single lines that exceed the limit
		if lineTokens > effectiveMax {
			// Flush current chunk first if not empty
			if len(currentLines) > 0 {
				c := Chunk{
					Content:     strings.Join(currentLines, "\n"),
					StartLine:   startLine,
					EndLine:     chunk.StartLine + i - 1,
					FilePath:    chunk.FilePath,
					Description: chunk.Description + fmt.Sprintf(" (part %d)", len(chunks)+1),
				}
				chunks = append(chunks, c)
				currentLines = nil
				currentTokens = 0
			}

			// Split the long line into multiple chunks
			splitLines := splitLongLine(line, effectiveMax)
			for _, sl := range splitLines {
				c := Chunk{
					Content:     sl,
					StartLine:   chunk.StartLine + i,
					EndLine:     chunk.StartLine + i,
					FilePath:    chunk.FilePath,
					Description: chunk.Description + fmt.Sprintf(" (part %d)", len(chunks)+1),
				}
				chunks = append(chunks, c)
			}
			startLine = chunk.StartLine + i + 1
			continue
		}

		nextTokens := lineTokens
		if len(groupCosts) > 0 && groupCosts[i] > nextTokens {
			nextTokens = groupCosts[i]
		}
		if currentTokens+nextTokens > effectiveMax && len(currentLines) > 0 {
			c := Chunk{
				Content:     strings.Join(currentLines, "\n"),
				StartLine:   startLine,
				EndLine:     chunk.StartLine + i - 1,
				FilePath:    chunk.FilePath,
				Description: chunk.Description + fmt.Sprintf(" (part %d)", len(chunks)+1),
			}
			chunks = append(chunks, c)

			currentLines = nil
			currentTokens = 0
			startLine = chunk.StartLine + i
		}

		currentLines = append(currentLines, line)
		currentTokens += lineTokens
	}

	if len(currentLines) > 0 {
		c := Chunk{
			Content:     strings.Join(currentLines, "\n"),
			StartLine:   startLine,
			EndLine:     chunk.EndLine,
			FilePath:    chunk.FilePath,
			Description: chunk.Description,
		}
		if len(chunks) > 0 {
			c.Description += fmt.Sprintf(" (part %d)", len(chunks)+1)
		}
		chunks = append(chunks, c)
	}

	return chunks
}

// splitLongLine splits a single line that exceeds the token limit at word boundaries.
func splitLongLine(line string, maxTokens int) []string {
	words := strings.Fields(line)
	if len(words) == 0 {
		return []string{line}
	}

	var result []string
	var current strings.Builder
	currentTokens := 0

	for _, word := range words {
		wordTokens := estimateTokens(word + " ")

		// If a single word exceeds the limit, include it anyway (unavoidable)
		if wordTokens > maxTokens && current.Len() == 0 {
			result = append(result, word)
			continue
		}

		if currentTokens+wordTokens > maxTokens && current.Len() > 0 {
			result = append(result, strings.TrimSpace(current.String()))
			current.Reset()
			currentTokens = 0
		}

		current.WriteString(word)
		current.WriteString(" ")
		currentTokens += wordTokens
	}

	if current.Len() > 0 {
		result = append(result, strings.TrimSpace(current.String()))
	}

	return result
}

func buildFuncDescription(fileName, pkgName string, fn *ast.FuncDecl) string {
	var b strings.Builder
	b.WriteString("Go function ")

	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		recv := fn.Recv.List[0]
		b.WriteString("(")
		b.WriteString(formatType(recv.Type))
		b.WriteString(").")
	}

	b.WriteString(fn.Name.Name)
	b.WriteString(" in package ")
	b.WriteString(pkgName)
	b.WriteString(" (")
	b.WriteString(fileName)
	b.WriteString(")")

	if fn.Doc != nil {
		doc := strings.TrimSpace(fn.Doc.Text())
		if doc != "" && len(doc) < 200 {
			b.WriteString(". ")
			b.WriteString(doc)
		}
	}

	return b.String()
}

func buildTypeDescription(fileName, pkgName string, ts *ast.TypeSpec) string {
	var b strings.Builder

	kind := "type"
	switch ts.Type.(type) {
	case *ast.StructType:
		kind = "struct"
	case *ast.InterfaceType:
		kind = "interface"
	}

	b.WriteString("Go ")
	b.WriteString(kind)
	b.WriteString(" ")
	b.WriteString(ts.Name.Name)
	b.WriteString(" in package ")
	b.WriteString(pkgName)
	b.WriteString(" (")
	b.WriteString(fileName)
	b.WriteString(")")

	return b.String()
}

func buildSizeDescription(path string, startLine, endLine int) string {
	return fmt.Sprintf("Code from %s (lines %d-%d)", filepath.Base(path), startLine, endLine)
}

func formatType(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + formatType(t.X)
	case *ast.SelectorExpr:
		return formatType(t.X) + "." + t.Sel.Name
	default:
		return "T"
	}
}

// EstimateTokens conservatively estimates model tokens for source code.
func EstimateTokens(text string) int {
	return modelcfg.EstimateTokens(text)
}

// estimateTokens is an internal alias for EstimateTokens.
func estimateTokens(text string) int {
	return EstimateTokens(text)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
