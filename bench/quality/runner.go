package quality

import (
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Runner executes quality evaluations against a codebase.
type Runner struct {
	codebase string
	topK     int
}

// NewRunner creates a new quality evaluation runner.
func NewRunner(codebase string, topK int) *Runner {
	return &Runner{
		codebase: codebase,
		topK:     topK,
	}
}

// RunSgrep executes sgrep and returns results with timing.
func (r *Runner) RunSgrep(ctx context.Context, query string) ([]SearchResult, time.Duration, error) {
	start := time.Now()

	cmd := exec.CommandContext(ctx, "sgrep", "-n", intToStr(r.topK), query)
	cmd.Dir = r.codebase
	output, err := cmd.CombinedOutput()

	elapsed := time.Since(start)

	if err != nil {
		// sgrep may return non-zero for no results, check if we have output
		if len(output) == 0 {
			return nil, elapsed, err
		}
	}

	results := parseSearchOutput(string(output))
	return results, elapsed, nil
}

// RunRipgrep executes ripgrep with multiple patterns (simulating agent behavior).
func (r *Runner) RunRipgrep(ctx context.Context, patterns []string) ([]SearchResult, time.Duration, int, error) {
	start := time.Now()

	seen := make(map[string]bool)
	var results []SearchResult
	attempts := 0

	for _, pattern := range patterns {
		attempts++

		cmd := exec.CommandContext(ctx, "rg", "-l", "--max-count", "10", pattern)
		cmd.Dir = r.codebase
		output, _ := cmd.CombinedOutput()

		for _, line := range strings.Split(string(output), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			file := filepath.Base(line)
			if !seen[file] {
				seen[file] = true
				results = append(results, SearchResult{File: file})
			}
		}

		// Simulate agent stopping when enough results found
		if len(results) >= r.topK {
			break
		}
	}

	elapsed := time.Since(start)
	return results, elapsed, attempts, nil
}

// EvaluateQuery runs both tools and computes metrics for a single query.
func (r *Runner) EvaluateQuery(ctx context.Context, query *QueryCase) (sgrepEval, rgEval *EvalResult, err error) {
	// Run sgrep
	sgrepResults, sgrepLatency, err := r.RunSgrep(ctx, query.Query)
	if err != nil {
		// Continue even on error to get partial results
		sgrepResults = nil
	}

	sgrepEval = ComputeAllMetrics(sgrepResults, query)
	sgrepEval.Tool = "sgrep"
	sgrepEval.LatencyMs = float64(sgrepLatency.Milliseconds())
	sgrepEval.TokensUsed = estimateTokens(query.Query) + estimateOutputTokens(sgrepResults)

	// Run ripgrep if patterns are provided
	if len(query.GrepPatterns) > 0 {
		rgResults, rgLatency, attempts, _ := r.RunRipgrep(ctx, query.GrepPatterns)
		rgEval = ComputeAllMetrics(rgResults, query)
		rgEval.Tool = "ripgrep"
		rgEval.LatencyMs = float64(rgLatency.Milliseconds())
		rgEval.TokensUsed = attempts*estimateTokens("rg pattern") + estimateOutputTokens(rgResults)
	}

	return sgrepEval, rgEval, nil
}

// RunDataset evaluates all queries in a dataset.
func (r *Runner) RunDataset(ctx context.Context, ds *Dataset) (*Report, error) {
	report := &Report{
		Dataset:   ds.Name,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Tools:     []string{"sgrep", "ripgrep"},
	}

	var sgrepResults, rgResults []EvalResult

	for _, query := range ds.Queries {
		sgrepEval, rgEval, err := r.EvaluateQuery(ctx, &query)
		if err != nil {
			continue
		}

		if sgrepEval != nil {
			report.Results = append(report.Results, *sgrepEval)
			sgrepResults = append(sgrepResults, *sgrepEval)
		}

		if rgEval != nil {
			report.Results = append(report.Results, *rgEval)
			rgResults = append(rgResults, *rgEval)
		}
	}

	// Compute summaries
	if len(sgrepResults) > 0 {
		report.Summaries = append(report.Summaries, AggregateSummary("sgrep", sgrepResults))
	}
	if len(rgResults) > 0 {
		report.Summaries = append(report.Summaries, AggregateSummary("ripgrep", rgResults))
	}

	return report, nil
}

// parseSearchOutput parses sgrep output into SearchResult slice.
// Handles multiple output formats:
//   - Standard: "file:line: content"
//   - Quiet: "file:startLine-endLine"
//   - JSON: [{"file":"...", ...}]
func parseSearchOutput(output string) []SearchResult {
	output = strings.TrimSpace(output)
	if output == "" {
		return nil
	}

	// Try JSON parsing first (if output starts with '[')
	if strings.HasPrefix(output, "[") {
		return parseJSONOutput(output)
	}

	var results []SearchResult
	seen := make(map[string]bool)

	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		file := extractFilePath(line)
		if file == "" {
			continue
		}

		// Deduplicate by file basename
		base := filepath.Base(file)
		if !seen[base] {
			seen[base] = true
			results = append(results, SearchResult{File: base})
		}
	}

	return results
}

// parseJSONOutput parses JSON array output from sgrep --json.
func parseJSONOutput(output string) []SearchResult {
	var raw []struct {
		File  string `json:"file"`
		Start int    `json:"start"`
		End   int    `json:"end"`
		Score float64 `json:"score"`
	}

	if err := json.Unmarshal([]byte(output), &raw); err != nil {
		return nil
	}

	seen := make(map[string]bool)
	var results []SearchResult

	for _, r := range raw {
		base := filepath.Base(r.File)
		if !seen[base] {
			seen[base] = true
			results = append(results, SearchResult{
				File:      base,
				Score:     r.Score,
				StartLine: r.Start,
				EndLine:   r.End,
			})
		}
	}

	return results
}

// extractFilePath extracts the file path from a line of sgrep output.
// Handles various formats:
//   - "file:123: content"
//   - "file:10-20"
//   - "C:\path\file:123" (Windows)
func extractFilePath(line string) string {
	// Handle Windows drive letters (e.g., C:\path\file:123)
	if len(line) >= 2 && line[1] == ':' && isLetter(line[0]) {
		// Windows path: find the next colon after drive letter
		rest := line[2:]
		if idx := strings.Index(rest, ":"); idx > 0 {
			return line[:2+idx]
		}
		return line
	}

	// Unix-style path: split on first colon
	if idx := strings.Index(line, ":"); idx > 0 {
		return line[:idx]
	}

	// No colon found, treat entire line as file path
	return line
}

func isLetter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// estimateTokens estimates token count for a string (rough approximation).
func estimateTokens(text string) int {
	words := len(strings.Fields(text))
	return int(float64(words) * 1.3)
}

// estimateOutputTokens estimates tokens for search results.
func estimateOutputTokens(results []SearchResult) int {
	tokens := 0
	for _, r := range results {
		tokens += estimateTokens(r.File) + 5 // file path + metadata
	}
	return tokens
}

func intToStr(n int) string {
	if n < 0 {
		return "-" + intToStr(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return intToStr(n/10) + string(rune('0'+n%10))
}
