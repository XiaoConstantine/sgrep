package cli

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/sgrep/pkg/index"
	"github.com/XiaoConstantine/sgrep/pkg/rerank"
	"github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/server"
	"github.com/XiaoConstantine/sgrep/pkg/store"
	"github.com/XiaoConstantine/sgrep/pkg/util"
	claudeskill "github.com/XiaoConstantine/sgrep/plugins/sgrep/skills/sgrep"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	limit                   int
	showContext             bool
	jsonOutput              bool
	quiet                   bool
	threshold               float64
	includeTests            bool
	allChunks               bool
	hybridSearch            bool
	semanticWeight          float64
	bm25Weight              float64
	enableRerank            bool
	enableColBERT           bool
	colbertAdaptiveSegments bool
	rerankTopK              int

	// Index flags
	indexWorkers                 int
	indexQuantize                string
	indexSQLVectors              bool
	indexColBERTPreindex         bool
	indexColBERTAdaptiveSegments bool
	indexColBERTCodec            string

	// Debug flags
	debugLevel   int    // 0=off, 1=summary, 2=detailed (set via -d count)
	debugLogFile string // optional log file path
	enableTrace  bool   // enable FlightRecorder tracing

	// Setup flags
	setupWithRerank bool
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
	Args:              cobra.MaximumNArgs(1),
	PersistentPreRunE: setupDebug,
	RunE:              runSearch,
}

// setupDebug configures debug output based on flags.
func setupDebug(cmd *cobra.Command, args []string) error {
	// Set debug level
	util.SetDebugLevel(util.DebugLevel(debugLevel))

	// Set up debug writer
	var writer io.Writer = os.Stderr
	if debugLogFile != "" {
		f, err := os.OpenFile(debugLogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return fmt.Errorf("failed to open debug log file: %w", err)
		}
		// Note: We don't close this file - it stays open for the duration of the command
		// This is acceptable for CLI tools
		writer = io.MultiWriter(os.Stderr, f)
	}
	util.SetDebugWriter(writer)

	// Start FlightRecorder if tracing enabled
	if enableTrace {
		if err := util.StartGlobalRecorder(); err != nil {
			return fmt.Errorf("failed to start flight recorder: %w", err)
		}
		util.Debugf(util.DebugSummary, "FlightRecorder tracing enabled")
	}

	return nil
}

func init() {
	// Debug flags (persistent - available to all commands)
	rootCmd.PersistentFlags().CountVarP(&debugLevel, "debug", "d",
		"Debug level: -d (summary timing), -dd (detailed per-operation timing)")
	rootCmd.PersistentFlags().StringVar(&debugLogFile, "debug-log", "",
		"Write debug output to file (in addition to stderr)")
	rootCmd.PersistentFlags().BoolVar(&enableTrace, "trace", false,
		"Enable FlightRecorder tracing (auto-captures slow queries to ~/.sgrep/traces/)")

	// Search flags
	rootCmd.Flags().IntVarP(&limit, "limit", "n", 10, "Maximum number of results")
	rootCmd.Flags().BoolVarP(&showContext, "context", "c", false, "Show code context")
	rootCmd.Flags().BoolVar(&jsonOutput, "json", false, "Output as JSON (for agents)")
	rootCmd.Flags().BoolVarP(&quiet, "quiet", "q", false, "Minimal output (paths only)")
	rootCmd.Flags().Float64Var(&threshold, "threshold", 1.5, "Similarity threshold (cosine distance, lower is more similar)")
	rootCmd.Flags().BoolVarP(&includeTests, "include-tests", "t", false, "Include test files in results")
	rootCmd.Flags().BoolVar(&allChunks, "all-chunks", false, "Show all matching chunks (disable deduplication)")
	rootCmd.Flags().BoolVar(&hybridSearch, "hybrid", false, "Enable hybrid search (semantic + BM25)")
	rootCmd.Flags().Float64Var(&semanticWeight, "semantic-weight", 0.6, "Weight for semantic score in hybrid mode")
	rootCmd.Flags().Float64Var(&bm25Weight, "bm25-weight", 0.4, "Weight for BM25 score in hybrid mode")
	rootCmd.Flags().BoolVar(&enableRerank, "rerank", false, "Enable cross-encoder reranking (requires reranker model)")
	rootCmd.Flags().BoolVar(&enableColBERT, "colbert", false, "Enable ColBERT late interaction scoring (no extra model needed)")
	rootCmd.Flags().BoolVar(&colbertAdaptiveSegments, "colbert-adaptive-segments", false, "Use adaptive sqrt(M) segment budgets for ColBERT scoring (experimental)")
	rootCmd.Flags().IntVar(&rerankTopK, "rerank-topk", 50, "Number of candidates to fetch for reranking")

	// Add subcommands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(watchCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(clearCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(serverCmd)
	rootCmd.AddCommand(installClaudeCodeCmd)
	rootCmd.AddCommand(installSkillCmd)
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

	// Open store with adaptive search mode
	s, err := store.OpenForSearch(indexPath)
	if err != nil {
		return fmt.Errorf("failed to open index: %w", err)
	}
	defer func() { _ = s.Close() }()

	// Ensure FTS5 index exists for hybrid search
	if hybridSearch {
		if err := store.EnsureFTS5IfNeeded(s); err != nil {
			return fmt.Errorf("failed to initialize FTS5 for hybrid search: %w", err)
		}
	}

	// Build search options
	opts := search.DefaultSearchOptions()
	opts.Limit = limit
	opts.Threshold = threshold
	opts.IncludeTests = includeTests
	opts.Deduplicate = !allChunks
	opts.UseHybrid = hybridSearch
	opts.SemanticWeight = semanticWeight
	opts.BM25Weight = bm25Weight
	opts.UseRerank = enableRerank
	opts.RerankTopK = rerankTopK

	// Create searcher config
	searchCfg := search.Config{
		Store:            s,
		AdaptiveSegments: colbertAdaptiveSegments,
	}

	// Auto-enable ColBERT if MMap segment store exists (fast pre-computed segments)
	// Can be explicitly enabled via --colbert or automatically with --rerank
	indexDir := filepath.Dir(indexPath)
	mmapPath := filepath.Join(indexDir, "colbert_segments.mmap")
	if _, err := os.Stat(mmapPath); err == nil {
		mmapStore, err := store.OpenMMapSegmentStore(indexDir, 768) // 768 dims for nomic-embed
		if err == nil {
			searchCfg.SegmentStore = mmapStore
			defer func() { _ = mmapStore.Close() }()
			// Auto-enable ColBERT when MMap is available (fast pre-computed segments)
			if !enableColBERT {
				enableColBERT = true
				util.Debugf(util.DebugSummary, "Auto-enabled ColBERT (MMap segments available)")
			}
			util.Debugf(util.DebugSummary, "Using MMap segment store for ColBERT")
		}
	}
	opts.UseColBERT = enableColBERT || enableRerank

	// Set up reranker if enabled
	if enableRerank {
		// Check if reranker model is available
		if !rerank.RerankerAvailable() {
			return fmt.Errorf("reranker model not found. Run 'sgrep setup --with-rerank' first")
		}

		reranker, err := rerank.New()
		if err != nil {
			return fmt.Errorf("failed to initialize reranker: %w", err)
		}
		searchCfg.Reranker = reranker
	}

	// Search
	searcher := search.NewWithConfig(searchCfg)
	results, err := searcher.SearchWithOptions(ctx, query, opts)
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
	if resolved, err := filepath.EvalSymlinks(abs); err == nil && resolved != abs {
		repoID = hashPath(resolved)
		indexPath = filepath.Join(sgrepHome, "repos", repoID, "index.db")
		if _, err := os.Stat(indexPath); err == nil {
			return indexPath, nil
		}
	}
	if indexPath, ok := findIndexDirByMetadata(sgrepHome, abs); ok {
		return indexPath, nil
	}

	return "", fmt.Errorf("no index found for %s. Run 'sgrep index .' first", abs)
}

func findIndexDirByMetadata(sgrepHome, targetPath string) (string, bool) {
	reposDir := filepath.Join(sgrepHome, "repos")
	entries, err := os.ReadDir(reposDir)
	if err != nil {
		return "", false
	}
	targetCanonical := canonicalDiskPath(targetPath)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		repoDir := filepath.Join(reposDir, entry.Name())
		metadataPath := filepath.Join(repoDir, "metadata.json")
		data, err := os.ReadFile(metadataPath)
		if err != nil {
			continue
		}
		var metadata struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal(data, &metadata); err != nil || metadata.Path == "" {
			continue
		}
		if canonicalDiskPath(metadata.Path) != targetCanonical {
			continue
		}
		indexPath := filepath.Join(repoDir, "index.db")
		if _, err := os.Stat(indexPath); err == nil {
			return indexPath, true
		}
	}
	return "", false
}

func canonicalDiskPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	}
	return filepath.Clean(abs)
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

		// Build config from flags
		cfg := index.DefaultIndexConfig()
		if indexWorkers > 0 {
			cfg.Workers = indexWorkers
		}
		if indexQuantize != "" {
			cfg.Quantization = store.ParseQuantizationMode(indexQuantize)
		}
		cfg.CompactVectorStorage = !indexSQLVectors
		cfg.AdaptiveColBERTSegments = indexColBERTAdaptiveSegments
		cfg.ColBERTCodec = store.ParseColBERTCodec(indexColBERTCodec)

		ctx := context.Background()
		indexer, err := index.NewWithConfig(path, cfg)
		if err != nil {
			return fmt.Errorf("failed to create indexer: %w", err)
		}
		defer func() { _ = indexer.Close() }()

		if err := indexer.Index(ctx); err != nil {
			return err
		}

		if indexer.CompactVectorStoreWritten() {
			chunkCount, fileCount := indexer.CompactVectorStoreCounts()
			fmt.Printf("Wrote compact TQ-MSE vector stores (%d chunk vectors, %d file vectors)\n", chunkCount, fileCount)
		} else {
			// Export vectors to compact TQ-MSE stores for faster, smaller semantic retrieval.
			fmt.Println("\nExporting vectors to compact TQ-MSE stores...")
			vecCount, err := indexer.RebuildTQVectorStore(ctx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to export vectors to compact TQ-MSE stores: %v\n", err)
			} else {
				fmt.Printf("Exported %d chunk vectors to compact TQ-MSE stores\n", vecCount)
			}
		}
		if err := os.Remove(filepath.Join(indexer.RepoDir(), "vectors.mmap")); err != nil && !os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Warning: failed to remove legacy vectors.mmap: %v\n", err)
		}

		// Pre-compute ColBERT segments if requested
		if indexColBERTPreindex {
			fmt.Println("\nPre-computing ColBERT segments for fast query-time scoring...")
			processed, err := indexer.ComputeColBERTSegments(ctx)
			if err != nil {
				return fmt.Errorf("failed to compute ColBERT segments: %w", err)
			}
			fmt.Printf("Computed segments for %d chunks\n", processed)

			// Export to MMap for faster query-time access
			fmt.Println("Exporting segments to MMap store...")
			segCount, err := indexer.ExportColBERTToMMap(ctx, indexer.RepoDir())
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to export to MMap: %v\n", err)
			} else {
				fmt.Printf("Exported %d segments to MMap store\n", segCount)
			}
		} else if err := os.Remove(filepath.Join(indexer.RepoDir(), "colbert_segments.mmap")); err != nil && !os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Warning: failed to remove stale colbert_segments.mmap: %v\n", err)
		}

		if err := indexer.Checkpoint(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to checkpoint index: %v\n", err)
		}

		return nil
	},
}

func init() {
	// Add index-specific flags
	indexCmd.Flags().IntVar(&indexWorkers, "workers", 0, "Number of parallel workers (default: 2x CPU cores, max 16)")
	indexCmd.Flags().StringVar(&indexQuantize, "quantize", "int8", "Quantization mode: none (4x size), int8 (1x size), binary (0.125x size)")
	indexCmd.Flags().BoolVar(&indexSQLVectors, "sql-vectors", false, "Also store full first-stage vectors in SQLite/libSQL for legacy vector search")
	indexCmd.Flags().BoolVar(&indexColBERTPreindex, "colbert-preindex", true, "Pre-compute ColBERT segment embeddings for fast query-time scoring (default: true)")
	indexCmd.Flags().BoolVar(&indexColBERTAdaptiveSegments, "colbert-adaptive-segments", false, "Use adaptive sqrt(M) segment budgets during ColBERT preindexing (experimental)")
	indexCmd.Flags().StringVar(&indexColBERTCodec, "colbert-codec", "", "ColBERT segment codec: tqmse, int8, or pq6 (default: reuse existing repo codec, otherwise tqmse)")
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
		cfg := index.DefaultIndexConfig()
		cfg.CompactVectorStorage = false
		indexer, err := index.NewWithConfig(path, cfg)
		if err != nil {
			return fmt.Errorf("failed to create indexer: %w", err)
		}
		defer func() { _ = indexer.Close() }()

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

		s, err := store.OpenForStats(indexPath)
		if err != nil {
			return err
		}
		defer func() { _ = s.Close() }()

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

Use --with-rerank to also download the reranker model (~636MB) for better
search precision with the --rerank flag.

The models are stored in ~/.sgrep/models/`,
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr, err := server.NewManager()
		if err != nil {
			return err
		}

		if err := mgr.Setup(true); err != nil {
			return err
		}

		// Optionally download reranker model
		if setupWithRerank {
			fmt.Println()
			fmt.Println("Downloading reranker model (~636MB)...")

			rerankMgr, err := rerank.NewRerankerManager()
			if err != nil {
				return fmt.Errorf("failed to initialize reranker manager: %w", err)
			}

			if rerankMgr.ModelExists() {
				fmt.Printf("✓ Reranker model already downloaded: %s\n", rerankMgr.ModelPath())
			} else {
				lastPct := -1
				err = rerankMgr.DownloadModel(func(downloaded, total int64) {
					if total > 0 {
						pct := int(downloaded * 100 / total)
						if pct != lastPct && pct%10 == 0 {
							fmt.Printf("  %d%%\n", pct)
							lastPct = pct
						}
					}
				})
				if err != nil {
					return fmt.Errorf("failed to download reranker model: %w", err)
				}
				fmt.Printf("✓ Reranker model downloaded: %s\n", rerankMgr.ModelPath())
			}
		}

		return nil
	},
}

func init() {
	setupCmd.Flags().BoolVar(&setupWithRerank, "with-rerank", false, "Also download reranker model for --rerank flag")
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

		fmt.Println("=== Embedding Server ===")
		running, pid, port := mgr.Status()
		if running {
			fmt.Printf("Status: running on port %d", port)
			if pid > 0 {
				fmt.Printf(" (PID %d)", pid)
			}
			fmt.Println()
		} else {
			fmt.Println("Status: not running")
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

		// Show reranker status
		fmt.Println()
		fmt.Println("=== Reranker Server ===")
		rerankMgr, err := rerank.NewRerankerManager()
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			running, pid, port := rerankMgr.Status()
			if running {
				fmt.Printf("Status: running on port %d", port)
				if pid > 0 {
					fmt.Printf(" (PID %d)", pid)
				}
				fmt.Println()
			} else {
				fmt.Println("Status: not running")
			}

			if rerankMgr.ModelExists() {
				fmt.Printf("Model: %s\n", rerankMgr.ModelPath())
			} else {
				fmt.Println("Model: not downloaded (run 'sgrep setup --with-rerank')")
			}
		}

		return nil
	},
}

// Install Claude Code command
var installClaudeCodeCmd = &cobra.Command{
	Use:   "install-claude-code",
	Short: "Install sgrep plugin and Claude skill for Claude Code",
	Long: `Installs the sgrep plugin for Claude Code.

This creates the plugin in ~/.claude/plugins/sgrep and a Claude skill in ~/.claude/skills/sgrep/SKILL.md with:
- Auto-indexing on session start
- Watch mode for live index updates
- Standard Claude Code skill installation

After installation, restart Claude Code to activate the plugin.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return installClaudeCodePluginAndSkill()
	},
}

var installSkillCmd = &cobra.Command{
	Use:   "install-skill",
	Short: "Install the sgrep skill for shared agent skill directories",
	Long: `Installs the sgrep skill into the standard user-level skill locations.

This creates:
- ~/.agents/skills/sgrep/SKILL.md for cross-client interoperability
- ~/.claude/skills/sgrep/SKILL.md for Claude Code compatibility`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return installSharedSkills()
	},
}

func installClaudeCodePluginAndSkill() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	pluginDir := filepath.Join(homeDir, ".claude", "plugins", "sgrep")
	skillsDir := filepath.Join(homeDir, ".claude", "skills", "sgrep")
	pluginSkillPath := filepath.Join(pluginDir, "skills", "sgrep", "SKILL.md")

	// Create plugin directory structure and user-level skill directory.
	dirs := []string{
		pluginDir,
		filepath.Join(pluginDir, ".claude-plugin"),
		filepath.Join(pluginDir, "hooks"),
		filepath.Join(pluginDir, "skills", "sgrep"),
		skillsDir,
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	// Write plugin.json
	pluginJSON := `{
  "name": "sgrep",
  "description": "Smart semantic + hybrid code search",
  "version": "0.1.0",
  "author": {
    "name": "Xiao Cui"
  },
  "hooks": "./hooks/hook.json"
}
`
	if err := os.WriteFile(filepath.Join(pluginDir, ".claude-plugin", "plugin.json"), []byte(pluginJSON), 0644); err != nil {
		return fmt.Errorf("failed to write plugin.json: %w", err)
	}

	// Write hook.json
	hookJSON := `{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/sgrep_start.sh",
            "timeout": 30
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/sgrep_stop.sh",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
`
	if err := os.WriteFile(filepath.Join(pluginDir, "hooks", "hook.json"), []byte(hookJSON), 0644); err != nil {
		return fmt.Errorf("failed to write hook.json: %w", err)
	}

	// Write sgrep_start.sh
	startScript := `#!/bin/bash
# Start sgrep indexing and watch for the current project

# Check if sgrep is installed
if ! command -v sgrep &> /dev/null; then
    echo "sgrep not found. Install with: brew tap XiaoConstantine/tap && brew install sgrep"
    exit 0
fi

# Get the project root
PROJECT_ROOT="${CLAUDE_PROJECT_ROOT:-$(pwd)}"

# Check if already indexed, if not index first
if ! sgrep status &> /dev/null; then
    echo "sgrep: Indexing $PROJECT_ROOT..."
    sgrep index "$PROJECT_ROOT" &> /dev/null
fi

# Start watch mode in background (if not already running)
if ! pgrep -f "sgrep watch" > /dev/null; then
    nohup sgrep watch "$PROJECT_ROOT" &> /dev/null &
    echo "sgrep: Watch mode started"
fi
`
	if err := os.WriteFile(filepath.Join(pluginDir, "hooks", "sgrep_start.sh"), []byte(startScript), 0755); err != nil {
		return fmt.Errorf("failed to write sgrep_start.sh: %w", err)
	}

	// Write sgrep_stop.sh
	stopScript := `#!/bin/bash
# Stop sgrep watch mode

if pgrep -f "sgrep watch" > /dev/null; then
    pkill -f "sgrep watch"
    echo "sgrep: Watch mode stopped"
fi
`
	if err := os.WriteFile(filepath.Join(pluginDir, "hooks", "sgrep_stop.sh"), []byte(stopScript), 0755); err != nil {
		return fmt.Errorf("failed to write sgrep_stop.sh: %w", err)
	}

	// Write the canonical Claude skill shipped with the plugin.
	if err := os.WriteFile(pluginSkillPath, []byte(claudeskill.Content), 0644); err != nil {
		return fmt.Errorf("failed to write plugin skill: %w", err)
	}
	if err := copyDir(filepath.Dir(pluginSkillPath), skillsDir); err != nil {
		return fmt.Errorf("failed to install Claude skill: %w", err)
	}

	fmt.Println("✓ sgrep plugin installed for Claude Code")
	fmt.Printf("  Plugin: %s\n", pluginDir)
	fmt.Printf("  Skill:  %s\n", filepath.Join(skillsDir, "SKILL.md"))
	fmt.Println()
	fmt.Println("Restart Claude Code to activate the plugin.")
	fmt.Println()
	fmt.Println("The plugin will automatically:")
	fmt.Println("  • Index your project on session start")
	fmt.Println("  • Keep the index updated via watch mode")
	fmt.Println("  • Install the 'sgrep' Claude skill in ~/.claude/skills/")

	return nil
}

func installSharedSkills() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	canonicalSkillDir := filepath.Join(homeDir, ".claude", "plugins", "sgrep", "skills", "sgrep")
	if _, err := os.Stat(filepath.Join(canonicalSkillDir, "SKILL.md")); err != nil {
		// Fall back to the skill embedded in the binary if the plugin is not installed.
		canonicalSkillDir = ""
	}

	skillDirs := []string{
		filepath.Join(homeDir, ".agents", "skills", "sgrep"),
		filepath.Join(homeDir, ".claude", "skills", "sgrep"),
	}
	for _, dir := range skillDirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	if canonicalSkillDir != "" {
		for _, dir := range skillDirs {
			if err := copyDir(canonicalSkillDir, dir); err != nil {
				return fmt.Errorf("failed to install skill into %s: %w", dir, err)
			}
		}
	} else {
		for _, dir := range skillDirs {
			if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(claudeskill.Content), 0644); err != nil {
				return fmt.Errorf("failed to write skill into %s: %w", dir, err)
			}
		}
	}

	fmt.Println("✓ sgrep skill installed")
	fmt.Printf("  Skill: %s\n", filepath.Join(homeDir, ".agents", "skills", "sgrep", "SKILL.md"))
	fmt.Printf("  Skill: %s\n", filepath.Join(homeDir, ".claude", "skills", "sgrep", "SKILL.md"))
	fmt.Println()
	fmt.Println("The skill is now available in the shared cross-client path and the Claude-compatible path.")

	return nil
}

func copyDir(srcDir, dstDir string) error {
	return filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(dstDir, rel)

		if d.IsDir() {
			return os.MkdirAll(targetPath, 0755)
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		return os.WriteFile(targetPath, data, info.Mode().Perm())
	})
}
