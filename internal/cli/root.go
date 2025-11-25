package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/sgrep/internal/index"
	"github.com/XiaoConstantine/sgrep/internal/search"
	"github.com/XiaoConstantine/sgrep/internal/store"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	limit       int
	showContext bool
	jsonOutput  bool
	quiet       bool
	threshold   float64
)

func Execute() error {
	return rootCmd.Execute()
}

var rootCmd = &cobra.Command{
	Use:   "sgrep [query]",
	Short: "Semantic grep for code - find code by intent, not exact patterns",
	Long: `sgrep is a semantic code search tool that understands what you mean.

Designed to complement ripgrep (exact text) and ast-grep (AST patterns):
  - ripgrep: "findUser" → exact string match
  - ast-grep: $fn($args) → structural pattern
  - sgrep: "user authentication" → semantic intent

Optimized for coding agents (Amp, Claude Code) with minimal token output.`,
	Args: cobra.MaximumNArgs(1),
	RunE: runSearch,
}

func init() {
	// Search flags
	rootCmd.Flags().IntVarP(&limit, "limit", "n", 10, "Maximum number of results")
	rootCmd.Flags().BoolVarP(&showContext, "context", "c", false, "Show code context")
	rootCmd.Flags().BoolVar(&jsonOutput, "json", false, "Output as JSON (for agents)")
	rootCmd.Flags().BoolVarP(&quiet, "quiet", "q", false, "Minimal output (paths only)")
	rootCmd.Flags().Float64Var(&threshold, "threshold", 0.5, "Similarity threshold (0-1, lower is more similar)")

	// Add subcommands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(watchCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(clearCmd)
}

func runSearch(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmd.Help()
	}

	query := args[0]
	ctx := context.Background()

	// Find index directory
	indexPath, err := findIndexDir(".")
	if err != nil {
		return fmt.Errorf("no index found. Run 'sgrep index .' first")
	}

	// Open store
	s, err := store.Open(indexPath)
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer s.Close()

	// Search
	searcher := search.New(s)
	results, err := searcher.Search(ctx, query, limit, threshold)
	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	// Output results
	return outputResults(results, showContext, jsonOutput, quiet)
}

func outputResults(results []search.Result, showContext, jsonOut, quiet bool) error {
	if len(results) == 0 {
		if !quiet {
			fmt.Println("No results found")
		}
		return nil
	}

	if jsonOut {
		return json.NewEncoder(os.Stdout).Encode(results)
	}

	for _, r := range results {
		if quiet {
			// Minimal: just file:lines
			fmt.Printf("%s:%d-%d\n", r.FilePath, r.StartLine, r.EndLine)
		} else if showContext {
			// With context
			fmt.Printf("%s:%d-%d (%.2f)\n", r.FilePath, r.StartLine, r.EndLine, r.Score)
			// Indent and truncate content
			lines := strings.Split(r.Content, "\n")
			maxLines := 5
			if len(lines) > maxLines {
				lines = lines[:maxLines]
			}
			for _, line := range lines {
				if len(line) > 80 {
					line = line[:77] + "..."
				}
				fmt.Printf("  %s\n", line)
			}
			fmt.Println()
		} else {
			// Default: file:lines
			fmt.Printf("%s:%d-%d\n", r.FilePath, r.StartLine, r.EndLine)
		}
	}

	return nil
}

func findIndexDir(startPath string) (string, error) {
	abs, err := filepath.Abs(startPath)
	if err != nil {
		return "", err
	}

	for {
		indexPath := filepath.Join(abs, ".sgrep", "index.db")
		if _, err := os.Stat(indexPath); err == nil {
			return indexPath, nil
		}

		parent := filepath.Dir(abs)
		if parent == abs {
			break
		}
		abs = parent
	}

	return "", fmt.Errorf("no .sgrep directory found")
}

// Index command
var indexCmd = &cobra.Command{
	Use:   "index [path]",
	Short: "Index a directory for semantic search",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		path := "."
		if len(args) > 0 {
			path = args[0]
		}

		ctx := context.Background()
		indexer, err := index.New(path)
		if err != nil {
			return fmt.Errorf("failed to create indexer: %w", err)
		}
		defer indexer.Close()

		return indexer.Index(ctx)
	},
}

// Watch command
var watchCmd = &cobra.Command{
	Use:   "watch [path]",
	Short: "Watch directory and auto-index changes",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		path := "."
		if len(args) > 0 {
			path = args[0]
		}

		ctx := context.Background()
		indexer, err := index.New(path)
		if err != nil {
			return fmt.Errorf("failed to create indexer: %w", err)
		}
		defer indexer.Close()

		return indexer.Watch(ctx)
	},
}

// Status command
var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index status",
	RunE: func(cmd *cobra.Command, args []string) error {
		indexPath, err := findIndexDir(".")
		if err != nil {
			fmt.Println("No index found")
			return nil
		}

		s, err := store.Open(indexPath)
		if err != nil {
			return err
		}
		defer s.Close()

		stats, err := s.Stats(context.Background())
		if err != nil {
			return err
		}

		fmt.Printf("Index: %s\n", indexPath)
		fmt.Printf("Documents: %d\n", stats.Documents)
		fmt.Printf("Chunks: %d\n", stats.Chunks)
		fmt.Printf("Size: %s\n", formatBytes(stats.SizeBytes))
		return nil
	},
}

// Clear command
var clearCmd = &cobra.Command{
	Use:   "clear",
	Short: "Clear the index",
	RunE: func(cmd *cobra.Command, args []string) error {
		indexPath, err := findIndexDir(".")
		if err != nil {
			return nil // Already clear
		}

		dir := filepath.Dir(indexPath)
		if err := os.RemoveAll(dir); err != nil {
			return fmt.Errorf("failed to clear index: %w", err)
		}

		fmt.Println("Index cleared")
		return nil
	},
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}
