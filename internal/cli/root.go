package cli

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/sgrep/internal/index"
	"github.com/XiaoConstantine/sgrep/internal/search"
	"github.com/XiaoConstantine/sgrep/internal/server"
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
	rootCmd.Flags().Float64Var(&threshold, "threshold", 1.5, "Similarity threshold (L2 distance, lower is more similar)")

	// Add subcommands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(watchCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(clearCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(serverCmd)
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

	// Open store (use in-memory for fast search)
	s, err := store.OpenInMem(indexPath)
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

	// Get sgrep home
	sgrepHome, err := getSgrepHome()
	if err != nil {
		return "", err
	}

	// Hash the path to find the repo directory
	repoID := hashPath(abs)
	indexPath := filepath.Join(sgrepHome, "repos", repoID, "index.db")

	if _, err := os.Stat(indexPath); err == nil {
		return indexPath, nil
	}

	return "", fmt.Errorf("no index found for %s. Run 'sgrep index .' first", abs)
}

// getSgrepHome returns the sgrep home directory (~/.sgrep).
func getSgrepHome() (string, error) {
	if home := os.Getenv("SGREP_HOME"); home != "" {
		return home, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(homeDir, ".sgrep"), nil
}

// hashPath creates a short hash of a path for directory naming.
func hashPath(path string) string {
	h := sha256.Sum256([]byte(path))
	return fmt.Sprintf("%x", h[:6])
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

		s, err := store.OpenInMem(indexPath)
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

// List command - show all indexed repos
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all indexed repositories",
	RunE: func(cmd *cobra.Command, args []string) error {
		sgrepHome, err := getSgrepHome()
		if err != nil {
			return err
		}

		reposDir := filepath.Join(sgrepHome, "repos")
		entries, err := os.ReadDir(reposDir)
		if err != nil {
			if os.IsNotExist(err) {
				fmt.Println("No repositories indexed yet")
				return nil
			}
			return err
		}

		if len(entries) == 0 {
			fmt.Println("No repositories indexed yet")
			return nil
		}

		fmt.Println("Indexed repositories:")
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}

			metadataPath := filepath.Join(reposDir, entry.Name(), "metadata.json")
			data, err := os.ReadFile(metadataPath)
			if err != nil {
				continue
			}

			var metadata map[string]interface{}
			if err := json.Unmarshal(data, &metadata); err != nil {
				continue
			}

			path, _ := metadata["path"].(string)
			indexedAt, _ := metadata["indexed_at"].(string)

			fmt.Printf("  %s\n", path)
			if indexedAt != "" {
				fmt.Printf("    indexed: %s\n", indexedAt)
			}
		}

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

// Setup command - download model and verify llama-server
var setupCmd = &cobra.Command{
	Use:   "setup",
	Short: "Download embedding model and verify llama-server installation",
	Long: `Setup downloads the nomic-embed-text embedding model (~130MB) and
verifies that llama-server is installed.

The model is stored in ~/.sgrep/models/`,
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr, err := server.NewManager()
		if err != nil {
			return err
		}

		return mgr.Setup(true)
	},
}

// Server command group
var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Manage the embedding server",
	Long: `Commands to manage the llama.cpp embedding server.

The server runs automatically when needed, but you can also control it manually.`,
}

func init() {
	serverCmd.AddCommand(serverStartCmd)
	serverCmd.AddCommand(serverStopCmd)
	serverCmd.AddCommand(serverStatusCmd)
}

var serverStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the embedding server",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr, err := server.NewManager()
		if err != nil {
			return err
		}

		if mgr.IsRunning() {
			fmt.Println("Server already running")
			return nil
		}

		fmt.Println("Starting embedding server...")
		if err := mgr.Start(); err != nil {
			return err
		}
		fmt.Println("Server started")
		return nil
	},
}

var serverStopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the embedding server",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr, err := server.NewManager()
		if err != nil {
			return err
		}

		if !mgr.IsRunning() {
			fmt.Println("Server not running")
			return nil
		}

		if err := mgr.Stop(); err != nil {
			return err
		}
		fmt.Println("Server stopped")
		return nil
	},
}

var serverStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show embedding server status",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr, err := server.NewManager()
		if err != nil {
			return err
		}

		running, pid, port := mgr.Status()
		if running {
			fmt.Printf("Server running on port %d", port)
			if pid > 0 {
				fmt.Printf(" (PID %d)", pid)
			}
			fmt.Println()
		} else {
			fmt.Println("Server not running")
		}

		if mgr.ModelExists() {
			fmt.Printf("Model: %s\n", mgr.ModelPath())
		} else {
			fmt.Println("Model: not downloaded (run 'sgrep setup')")
		}

		if mgr.LlamaServerInstalled() {
			fmt.Println("llama-server: installed")
		} else {
			fmt.Println("llama-server: not found (brew install llama.cpp)")
		}

		return nil
	},
}
