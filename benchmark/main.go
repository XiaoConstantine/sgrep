package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// TestCase represents a search benchmark case.
type TestCase struct {
	Intent       string   `json:"intent"`
	TargetFiles  []string `json:"target_files"`
	GrepPatterns []string `json:"grep_patterns"`
}

// Result holds benchmark results for a single test case.
type Result struct {
	Intent string `json:"intent"`

	// sgrep metrics
	SgrepTokens   int           `json:"sgrep_tokens"`
	SgrepAttempts int           `json:"sgrep_attempts"`
	SgrepLatency  time.Duration `json:"sgrep_latency_ms"`
	SgrepFound    []string      `json:"sgrep_found"`
	SgrepPrecision float64      `json:"sgrep_precision"`
	SgrepRecall    float64      `json:"sgrep_recall"`

	// grep metrics
	GrepTokens   int           `json:"grep_tokens"`
	GrepAttempts int           `json:"grep_attempts"`
	GrepLatency  time.Duration `json:"grep_latency_ms"`
	GrepFound    []string      `json:"grep_found"`
	GrepPrecision float64      `json:"grep_precision"`
	GrepRecall    float64      `json:"grep_recall"`

	// Comparison
	TokenSavings float64 `json:"token_savings_percent"`
	Winner       string  `json:"winner"`
}

// BenchmarkSuite holds all test cases and results.
type BenchmarkSuite struct {
	Codebase   string    `json:"codebase"`
	TestCases  []TestCase `json:"test_cases"`
	Results    []Result   `json:"results"`
	Summary    Summary    `json:"summary"`
}

type Summary struct {
	TotalCases       int     `json:"total_cases"`
	SgrepWins        int     `json:"sgrep_wins"`
	GrepWins         int     `json:"grep_wins"`
	AvgTokenSavings  float64 `json:"avg_token_savings_percent"`
	AvgSgrepLatency  float64 `json:"avg_sgrep_latency_ms"`
	AvgGrepLatency   float64 `json:"avg_grep_latency_ms"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run benchmark/main.go <codebase_path>")
		os.Exit(1)
	}

	codebasePath := os.Args[1]

	// Default test cases - can be loaded from YAML
	testCases := []TestCase{
		{
			Intent:       "How does authentication work?",
			TargetFiles:  []string{"auth/", "middleware", "jwt", "login"},
			GrepPatterns: []string{"auth", "authenticate", "Auth", "login", "session", "jwt", "token"},
		},
		{
			Intent:       "Where is error handling implemented?",
			TargetFiles:  []string{"error", "Error", "handler"},
			GrepPatterns: []string{"error", "Error", "err", "handle", "HandleError"},
		},
		{
			Intent:       "How are database connections managed?",
			TargetFiles:  []string{"db", "database", "sql", "pool", "connect"},
			GrepPatterns: []string{"database", "sql", "db", "connection", "pool", "Connect"},
		},
		{
			Intent:       "What's the caching strategy?",
			TargetFiles:  []string{"cache", "redis", "memory"},
			GrepPatterns: []string{"cache", "Cache", "redis", "ttl", "expire", "Get", "Set"},
		},
		{
			Intent:       "How does the API handle rate limiting?",
			TargetFiles:  []string{"rate", "limit", "throttle"},
			GrepPatterns: []string{"rate", "limit", "throttle", "RateLimit", "bucket"},
		},
	}

	suite := &BenchmarkSuite{
		Codebase:  codebasePath,
		TestCases: testCases,
	}

	ctx := context.Background()

	fmt.Printf("Running benchmark on: %s\n", codebasePath)
	fmt.Printf("Test cases: %d\n\n", len(testCases))

	for i, tc := range testCases {
		fmt.Printf("[%d/%d] %s\n", i+1, len(testCases), tc.Intent)

		result := runTestCase(ctx, codebasePath, tc)
		suite.Results = append(suite.Results, result)

		fmt.Printf("  sgrep: %d tokens, %d attempts, %.0fms\n",
			result.SgrepTokens, result.SgrepAttempts, float64(result.SgrepLatency.Milliseconds()))
		fmt.Printf("  grep:  %d tokens, %d attempts, %.0fms\n",
			result.GrepTokens, result.GrepAttempts, float64(result.GrepLatency.Milliseconds()))
		fmt.Printf("  winner: %s (%.0f%% token savings)\n\n",
			result.Winner, result.TokenSavings)
	}

	// Calculate summary
	suite.Summary = calculateSummary(suite.Results)

	// Output results
	fmt.Println("=== Summary ===")
	fmt.Printf("sgrep wins: %d/%d\n", suite.Summary.SgrepWins, suite.Summary.TotalCases)
	fmt.Printf("Average token savings: %.1f%%\n", suite.Summary.AvgTokenSavings)
	fmt.Printf("Average latency - sgrep: %.0fms, grep: %.0fms\n",
		suite.Summary.AvgSgrepLatency, suite.Summary.AvgGrepLatency)

	// Save to file
	data, _ := json.MarshalIndent(suite, "", "  ")
	_ = os.WriteFile("benchmark_results.json", data, 0644)
	fmt.Println("\nResults saved to benchmark_results.json")
}

func runTestCase(ctx context.Context, codebase string, tc TestCase) Result {
	result := Result{
		Intent: tc.Intent,
	}

	// Run sgrep
	sgrepStart := time.Now()
	sgrepOutput, _ := runSgrep(codebase, tc.Intent)
	result.SgrepLatency = time.Since(sgrepStart)
	result.SgrepAttempts = 1
	result.SgrepTokens = estimateTokens(tc.Intent) + estimateTokens(sgrepOutput)
	result.SgrepFound = extractFiles(sgrepOutput)
	result.SgrepPrecision, result.SgrepRecall = calculateMetrics(result.SgrepFound, tc.TargetFiles)

	// Run grep (simulate agent trying multiple patterns)
	grepStart := time.Now()
	var grepOutput strings.Builder
	grepAttempts := 0
	grepFound := make(map[string]bool)

	for _, pattern := range tc.GrepPatterns {
		grepAttempts++
		output, _ := runGrep(codebase, pattern)
		grepOutput.WriteString(output)

		// Extract found files
		for _, f := range extractFiles(output) {
			grepFound[f] = true
		}

		// Check if we found enough (simulating agent stopping)
		if len(grepFound) >= len(tc.TargetFiles) {
			break
		}
	}

	result.GrepLatency = time.Since(grepStart)
	result.GrepAttempts = grepAttempts
	result.GrepTokens = grepAttempts*estimateTokens("rg pattern") + estimateTokens(grepOutput.String())

	var grepFoundList []string
	for f := range grepFound {
		grepFoundList = append(grepFoundList, f)
	}
	result.GrepFound = grepFoundList
	result.GrepPrecision, result.GrepRecall = calculateMetrics(grepFoundList, tc.TargetFiles)

	// Determine winner
	if result.SgrepTokens < result.GrepTokens {
		result.Winner = "sgrep"
		result.TokenSavings = float64(result.GrepTokens-result.SgrepTokens) / float64(result.GrepTokens) * 100
	} else {
		result.Winner = "grep"
		result.TokenSavings = 0
	}

	return result
}

func runSgrep(codebase, query string) (string, error) {
	cmd := exec.Command("sgrep", "-n", "5", query)
	cmd.Dir = codebase
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func runGrep(codebase, pattern string) (string, error) {
	cmd := exec.Command("rg", "-l", "--max-count", "10", pattern)
	cmd.Dir = codebase
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func extractFiles(output string) []string {
	var files []string
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Extract file path (before :line number)
		if idx := strings.Index(line, ":"); idx > 0 {
			line = line[:idx]
		}
		files = append(files, filepath.Base(line))
	}
	return files
}

func calculateMetrics(found []string, targets []string) (precision, recall float64) {
	if len(found) == 0 {
		return 0, 0
	}

	relevant := 0
	for _, f := range found {
		for _, t := range targets {
			if strings.Contains(f, t) || strings.Contains(t, f) {
				relevant++
				break
			}
		}
	}

	precision = float64(relevant) / float64(len(found))
	recall = float64(relevant) / float64(len(targets))
	return
}

func estimateTokens(text string) int {
	words := len(strings.Fields(text))
	return int(float64(words) * 1.3)
}

func calculateSummary(results []Result) Summary {
	s := Summary{TotalCases: len(results)}

	var totalSavings, totalSgrepLatency, totalGrepLatency float64

	for _, r := range results {
		if r.Winner == "sgrep" {
			s.SgrepWins++
		} else {
			s.GrepWins++
		}
		totalSavings += r.TokenSavings
		totalSgrepLatency += float64(r.SgrepLatency.Milliseconds())
		totalGrepLatency += float64(r.GrepLatency.Milliseconds())
	}

	if len(results) > 0 {
		s.AvgTokenSavings = totalSavings / float64(len(results))
		s.AvgSgrepLatency = totalSgrepLatency / float64(len(results))
		s.AvgGrepLatency = totalGrepLatency / float64(len(results))
	}

	return s
}
