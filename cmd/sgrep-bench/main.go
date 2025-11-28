// sgrep-bench runs quality and performance benchmarks for sgrep.
//
// Usage:
//
//	sgrep-bench quality -codebase /path/to/repo -dataset bench/quality/dataset.json
//	sgrep-bench compare -codebase /path/to/repo -query "how does auth work"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/XiaoConstantine/sgrep/bench/quality"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "quality":
		runQualityBenchmark(os.Args[2:])
	case "compare":
		runQuickCompare(os.Args[2:])
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`sgrep-bench - Quality and performance benchmarks for sgrep

Usage:
  sgrep-bench <command> [flags]

Commands:
  quality   Run full quality evaluation against a dataset
  compare   Quick comparison of sgrep vs ripgrep for a single query

Examples:
  sgrep-bench quality -codebase /path/to/repo -dataset bench/quality/dataset.json
  sgrep-bench compare -codebase . -query "how does authentication work"`)
}

func runQualityBenchmark(args []string) {
	fs := flag.NewFlagSet("quality", flag.ExitOnError)
	codebase := fs.String("codebase", ".", "Path to codebase to evaluate")
	datasetPath := fs.String("dataset", "bench/quality/dataset.json", "Path to dataset JSON")
	outputPath := fs.String("output", "bench_report.json", "Output report path")
	topK := fs.Int("k", 10, "Number of results to retrieve")
	verbose := fs.Bool("v", false, "Verbose output")

	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	// Load dataset
	ds, err := quality.LoadDataset(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading dataset: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("=== sgrep Quality Benchmark ===\n")
	fmt.Printf("Codebase:  %s\n", *codebase)
	fmt.Printf("Dataset:   %s (%d queries)\n", ds.Name, len(ds.Queries))
	fmt.Printf("Top-K:     %d\n\n", *topK)

	runner := quality.NewRunner(*codebase, *topK)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	report, err := runner.RunDataset(ctx, ds)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running benchmark: %v\n", err)
		os.Exit(1)
	}

	// Print per-query results if verbose
	if *verbose {
		fmt.Println("--- Per-Query Results ---")
		for _, r := range report.Results {
			fmt.Printf("[%s] %s\n", r.Tool, r.Query)
			fmt.Printf("  MRR=%.3f NDCG@10=%.3f P@10=%.3f R@10=%.3f Latency=%0.fms\n",
				r.MRR, r.NDCG10, r.PrecisionAt10, r.RecallAt10, r.LatencyMs)
		}
		fmt.Println()
	}

	// Print summary
	fmt.Println("=== Summary ===")
	for _, s := range report.Summaries {
		fmt.Printf("\n%s (%d queries):\n", s.Tool, s.NumQueries)
		fmt.Printf("  Mean MRR:      %.3f\n", s.MeanMRR)
		fmt.Printf("  Mean NDCG@5:   %.3f\n", s.MeanNDCG5)
		fmt.Printf("  Mean NDCG@10:  %.3f\n", s.MeanNDCG10)
		fmt.Printf("  Mean MAP:      %.3f\n", s.MeanMAP)
		fmt.Printf("  Mean P@5:      %.3f\n", s.MeanP5)
		fmt.Printf("  Mean P@10:     %.3f\n", s.MeanP10)
		fmt.Printf("  Mean R@5:      %.3f\n", s.MeanR5)
		fmt.Printf("  Mean R@10:     %.3f\n", s.MeanR10)
		fmt.Printf("  Mean Latency:  %.0fms\n", s.MeanLatencyMs)
		fmt.Printf("  Total Tokens:  %d\n", s.TotalTokens)
	}

	// Compare tools if both present
	if len(report.Summaries) >= 2 {
		sgrep := report.Summaries[0]
		rg := report.Summaries[1]

		fmt.Println("\n=== Comparison ===")

		if sgrep.MeanMRR > rg.MeanMRR {
			fmt.Printf("MRR:    sgrep wins (+%.1f%%)\n", (sgrep.MeanMRR-rg.MeanMRR)*100)
		} else {
			fmt.Printf("MRR:    ripgrep wins (+%.1f%%)\n", (rg.MeanMRR-sgrep.MeanMRR)*100)
		}

		if sgrep.MeanNDCG10 > rg.MeanNDCG10 {
			fmt.Printf("NDCG:   sgrep wins (+%.1f%%)\n", (sgrep.MeanNDCG10-rg.MeanNDCG10)*100)
		} else {
			fmt.Printf("NDCG:   ripgrep wins (+%.1f%%)\n", (rg.MeanNDCG10-sgrep.MeanNDCG10)*100)
		}

		if sgrep.TotalTokens < rg.TotalTokens {
			savings := float64(rg.TotalTokens-sgrep.TotalTokens) / float64(rg.TotalTokens) * 100
			fmt.Printf("Tokens: sgrep uses %.0f%% fewer tokens\n", savings)
		}
	}

	// Save report
	if err := quality.SaveReport(*outputPath, report); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving report: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nReport saved to: %s\n", *outputPath)
}

func runQuickCompare(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	codebase := fs.String("codebase", ".", "Path to codebase")
	queryStr := fs.String("query", "", "Query to search for")
	topK := fs.Int("k", 5, "Number of results")

	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	if *queryStr == "" {
		fmt.Fprintln(os.Stderr, "Error: -query is required")
		os.Exit(1)
	}

	runner := quality.NewRunner(*codebase, *topK)
	ctx := context.Background()

	fmt.Printf("=== Quick Compare ===\n")
	fmt.Printf("Query: %q\n\n", *queryStr)

	// Run sgrep
	fmt.Println("--- sgrep (semantic) ---")
	sgrepResults, sgrepLatency, err := runner.RunSgrep(ctx, *queryStr)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for _, r := range sgrepResults {
			fmt.Printf("  %s\n", r.File)
		}
		fmt.Printf("\nLatency: %dms, Results: %d\n", sgrepLatency.Milliseconds(), len(sgrepResults))
	}

	// Run ripgrep with query words as patterns
	fmt.Println("\n--- ripgrep (lexical) ---")
	patterns := extractPatterns(*queryStr)
	rgResults, rgLatency, attempts, _ := runner.RunRipgrep(ctx, patterns)
	for _, r := range rgResults {
		fmt.Printf("  %s\n", r.File)
	}
	fmt.Printf("\nLatency: %dms, Attempts: %d, Results: %d\n",
		rgLatency.Milliseconds(), attempts, len(rgResults))

	// Summary
	fmt.Println("\n=== Comparison ===")
	sgrepTokens := estimateTokens(*queryStr) + len(sgrepResults)*10
	rgTokens := attempts*15 + len(rgResults)*10

	fmt.Printf("sgrep:   %d tokens (1 query)\n", sgrepTokens)
	fmt.Printf("ripgrep: %d tokens (%d queries)\n", rgTokens, attempts)

	if sgrepTokens < rgTokens {
		savings := float64(rgTokens-sgrepTokens) / float64(rgTokens) * 100
		fmt.Printf("Winner:  sgrep (%.0f%% fewer tokens)\n", savings)
	} else {
		fmt.Println("Winner:  ripgrep")
	}
}

func extractPatterns(query string) []string {
	// Simple tokenization - in practice you might use more sophisticated NLP
	words := []string{}
	current := ""

	for _, c := range query {
		if c == ' ' || c == '?' || c == '.' || c == ',' {
			if len(current) >= 3 { // Skip short words
				words = append(words, current)
			}
			current = ""
		} else {
			current += string(c)
		}
	}
	if len(current) >= 3 {
		words = append(words, current)
	}

	return words
}

func estimateTokens(text string) int {
	words := 0
	inWord := false
	for _, c := range text {
		if c == ' ' {
			inWord = false
		} else if !inWord {
			inWord = true
			words++
		}
	}
	return int(float64(words) * 1.3)
}

func printJSON(v interface{}) { //nolint:unused // Reserved for future use
	data, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(data))
}
