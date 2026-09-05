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
	"unicode"
	"unicode/utf8"

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
	var packedStart token.Pos
	previousSmall := false

	for _, decl := range file.Decls {
		start := decl.Pos()
		var description string
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Doc != nil {
				start = d.Doc.Pos()
			}
			description = buildFuncDescription(baseName, pkgName, d)
		case *ast.GenDecl:
			if d.Tok != token.TYPE && d.Tok != token.CONST && d.Tok != token.VAR {
				previousSmall = false
				continue
			}
			// Keep declarations whole, including grouped specs and their comments.
			// Slicing each type from the group doc duplicated earlier specs, while
			// slicing from TypeSpec.Pos lost spec docs and the declaration keyword.
			if d.Doc != nil {
				start = d.Doc.Pos()
			}
			description = fmt.Sprintf("Go %s declarations in package %s (%s)", d.Tok, pkgName, baseName)
			if d.Tok == token.TYPE && len(d.Specs) == 1 {
				description = buildTypeDescription(baseName, pkgName, d.Specs[0].(*ast.TypeSpec))
			}
		default:
			previousSmall = false
			continue
		}
		text := content[start-1 : decl.End()-1]
		small := estimateTokens(text) < 20
		if small && previousSmall {
			// Pack consecutive small declarations instead of dropping them or
			// producing a vector for each tiny symbol. Include the real intervening
			// source, not synthetic separators or repeated declaration headers.
			packed := content[packedStart-1 : decl.End()-1]
			desc := fmt.Sprintf("Go declarations in package %s (%s)", pkgName, baseName)
			if estimateTokens(desc+"\n\n"+packed)+10 <= cfg.MaxTokens {
				previous := &chunks[len(chunks)-1]
				previous.Content = packed
				previous.EndLine = fset.PositionFor(decl.End(), false).Line
				previous.Description = desc
				continue
			}
		}
		chunks = append(chunks, Chunk{
			Content: text, StartLine: fset.PositionFor(start, false).Line,
			EndLine: fset.PositionFor(decl.End(), false).Line, FilePath: path, Description: description,
		})
		packedStart, previousSmall = start, small
	}

	// If no AST chunks, fall back to size-based
	if len(chunks) == 0 {
		return chunkBySize(path, content, cfg)
	}

	// Split oversized chunks
	var finalChunks []Chunk
	for _, chunk := range chunks {
		if estimateTokens(chunk.Content) > cfg.MaxTokens {
			// Go's native AST path keeps its established line-based boundaries.
			// Embedding validation handles any remaining description overhead.
			subChunks := splitOversized(chunk, cfg, false)
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
// It keeps syntax groups together when they fit, while Go retains its established
// size-only rechunking with overlap. MaxTokens includes the description; an
// indivisible oversized word or description is preserved so the embedding
// validator can reject it explicitly.
func SplitChunk(chunk Chunk, cfg *Config) []Chunk {
	if strings.EqualFold(filepath.Ext(chunk.FilePath), ".go") {
		// Go declaration fragments previously fell back to chunkBySize because
		// they lack a package clause. Use that splitter directly: re-parsing a
		// complete Go fragment could instead discard package-level statements.
		sizeCfg := *cfg
		sizeCfg.MaxTokens = max(100, cfg.MaxTokens-estimateTokens(chunk.Description)-20)
		parts, _ := chunkBySize(chunk.FilePath, chunk.Content, &sizeCfg)
		var result []Chunk
		for i, part := range parts {
			part.Description = chunk.Description + fmt.Sprintf(" (part %d)", i+1)
			part.StartLine += chunk.StartLine - 1
			part.EndLine += chunk.StartLine - 1
			if estimateTokens(part.Description+"\n\n"+part.Content) > cfg.MaxTokens {
				// Keep legacy boundaries where safe, but never let overlap or
				// per-line rounding bypass the embedding budget.
				result = append(result, splitOversized(part, cfg, true)...)
			} else {
				result = append(result, part)
			}
		}
		return result
	}
	return splitOversized(chunk, cfg, true)
}

func splitOversized(chunk Chunk, cfg *Config, syntaxAware bool) []Chunk {
	lines := strings.Split(chunk.Content, "\n")
	var chunks []Chunk
	var currentLines []string
	currentTokens := 0
	startLine := chunk.StartLine

	// Reserve tokens for description overhead (description + "\n\n" separator)
	descTokens := estimateTokens(chunk.Description) + 10 // +10 buffer for separator and part suffix
	effectiveMax := cfg.MaxTokens - descTokens
	if !syntaxAware {
		effectiveMax = max(100, effectiveMax)
	} else if effectiveMax <= 0 {
		return []Chunk{chunk}
	}

	lineCosts := make([]int, len(lines))
	for i, line := range lines {
		lineCosts[i] = estimateTokens(line)
		if syntaxAware {
			// Include separators and per-line rounding: summing bare line
			// estimates can undercount the joined content.
			lineCosts[i] = estimateTokens(line+"\n") + 1
		}
	}
	var groupCosts []int
	if syntaxAware {
		groupCosts = treeSitterGroupTokens(chunk.FilePath, chunk.Content, lineCosts, effectiveMax)
	}

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

// splitLongLine returns contiguous source slices, never reconstructed words.
// Whitespace is source too (including inside literals). Boundaries on either
// side of whitespace allow long whitespace runs to split without cutting UTF-8.
// An indivisible oversized word is retained for the embedding validator.
func splitLongLine(line string, maxTokens int) []string {
	if line == "" {
		return []string{line}
	}
	var result []string
	start, end := 0, 0
	extend := func(next int) {
		if next == end {
			return
		}
		if end > start && estimateTokens(line[start:next]) > maxTokens {
			result = append(result, line[start:end])
			start = end
		}
		end = next
	}
	for i, r := range line {
		if unicode.IsSpace(r) {
			extend(i)
			extend(i + utf8.RuneLen(r))
		}
	}
	extend(len(line))
	result = append(result, line[start:end])
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
