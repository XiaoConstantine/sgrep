package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
	"github.com/XiaoConstantine/sgrep/pkg/conv/parser"
	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/spf13/cobra"
)

var (
	// Conv search flags
	convLimit          int
	convThreshold      float64
	convAgent          string
	convProject        string
	convSince          string
	convBefore         string
	convAfter          string
	convHybrid         bool
	convExact          bool
	convJSON           bool
	convQuiet          bool
	convVerbose        bool
	convFormat         string
	convNoColor        bool
	convInteractive    bool
	convSemanticWeight float64
	convBM25Weight     float64

	// Conv view flags
	convTurn int

	// Conv context flags
	convContextTurns   int
	convContextSummary bool
	convContextCopy    bool

	// Conv export flags
	convOutput string

	// Conv copy flags
	convCopyCodeOnly bool
	convCopyFull     bool

	// Conv index flags
	convIndexSource string
	convIndexForce  bool
	convIndexWatch  bool

	// Conv resume flags
	convResumeFrom   int
	convResumeDryRun bool
)

// Register parsers once at initialization
func init() {
	parser.RegisterClaude()
	parser.RegisterCodex()
	parser.RegisterCursor()
	parser.RegisterOpenCode()
	parser.RegisterPiMono()
}

var convCmd = &cobra.Command{
	Use:   "conv [search] <query>",
	Short: "Search and manage coding agent conversations",
	Long: `Search across conversations from Claude Code, Codex CLI, Cursor, OpenCode, and Pi.

Examples:
  # Basic semantic search
  sgrep conv "authentication"
  sgrep conv search "authentication"

  # Filter by agent and time
  sgrep conv "database migration" --agent claude --since 7d

  # Project-specific search
  sgrep conv "fix bug" --project payment-service

  # Hybrid search (semantic + keyword)
  sgrep conv "JWT refresh_token" --hybrid

  # JSON output for scripting
  sgrep conv "auth" --json -n 1

  # Index conversations
  sgrep conv index
  sgrep conv index --source claude
  sgrep conv index --source opencode
  sgrep conv index --source pi`,
	Args: cobra.MaximumNArgs(1),
	RunE: runConvSearch,
}

func init() {
	rootCmd.AddCommand(convCmd)

	addConvSearchFlags(convCmd)
	addConvSearchFlags(convSearchCmd)

	// Add subcommands
	convCmd.AddCommand(convSearchCmd)
	convCmd.AddCommand(convViewCmd)
	convCmd.AddCommand(convResumeCmd)
	convCmd.AddCommand(convContextCmd)
	convCmd.AddCommand(convExportCmd)
	convCmd.AddCommand(convCopyCmd)
	convCmd.AddCommand(convIndexCmd)
	convCmd.AddCommand(convStatusCmd)
}

// Subcommands

var convSearchCmd = &cobra.Command{
	Use:   "search <query>",
	Short: "Search conversations (default)",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvSearch,
}

var convViewCmd = &cobra.Command{
	Use:   "view <session_id>",
	Short: "View full conversation",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvView,
}

var convResumeCmd = &cobra.Command{
	Use:   "resume <session_id>",
	Short: "Resume conversation in original agent",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvResume,
}

var convContextCmd = &cobra.Command{
	Use:   "context <session_id>",
	Short: "Extract context for injection into new session",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvContext,
}

var convExportCmd = &cobra.Command{
	Use:   "export <session_id>",
	Short: "Export conversation to file",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvExport,
}

var convCopyCmd = &cobra.Command{
	Use:   "copy <session_id>",
	Short: "Copy conversation/turn to clipboard",
	Args:  cobra.ExactArgs(1),
	RunE:  runConvCopy,
}

var convIndexCmd = &cobra.Command{
	Use:   "index",
	Short: "Index conversation sources",
	RunE:  runConvIndex,
}

var convStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index status",
	RunE:  runConvStatus,
}

func init() {
	// View flags
	convViewCmd.Flags().IntVarP(&convTurn, "turn", "n", -1, "Jump to specific turn")
	convViewCmd.Flags().BoolVar(&convJSON, "json", false, "Output as JSON")

	// Resume flags
	convResumeCmd.Flags().IntVar(&convResumeFrom, "from", -1, "Resume from specific turn")
	convResumeCmd.Flags().BoolVar(&convResumeDryRun, "dry-run", false, "Show command without executing")

	// Context flags
	convContextCmd.Flags().IntVar(&convContextTurns, "turns", 5, "Include last N turns")
	convContextCmd.Flags().BoolVar(&convContextSummary, "summary", false, "Generate condensed summary")
	convContextCmd.Flags().StringVar(&convFormat, "format", "prompt", "Output: prompt, markdown, json")
	convContextCmd.Flags().BoolVar(&convContextCopy, "copy", false, "Copy to clipboard automatically")

	// Export flags
	convExportCmd.Flags().StringVarP(&convOutput, "output", "o", "", "Output file path (default: stdout)")
	convExportCmd.Flags().StringVar(&convFormat, "format", "markdown", "Format: markdown, json, html")

	// Copy flags
	convCopyCmd.Flags().IntVar(&convTurn, "turn", -1, "Copy specific turn only")
	convCopyCmd.Flags().BoolVar(&convCopyCodeOnly, "code-only", false, "Copy only code blocks")
	convCopyCmd.Flags().BoolVar(&convCopyFull, "full", false, "Copy full conversation")

	// Index flags
	convIndexCmd.Flags().StringVar(&convIndexSource, "source", "", "Index specific source: claude, codex, cursor, opencode, pi")
	convIndexCmd.Flags().BoolVar(&convIndexForce, "force", false, "Re-index all (ignore cache)")
	convIndexCmd.Flags().BoolVar(&convIndexWatch, "watch", false, "Watch for new conversations")
}

func addConvSearchFlags(cmd *cobra.Command) {
	// Search flags
	cmd.Flags().IntVarP(&convLimit, "limit", "n", 10, "Max results to return")
	cmd.Flags().Float64VarP(&convThreshold, "threshold", "T", 0.5, "Similarity threshold 0-1")
	cmd.Flags().BoolVar(&convHybrid, "hybrid", false, "Enable hybrid semantic+keyword search")
	cmd.Flags().BoolVar(&convExact, "exact", false, "Exact keyword match only (no semantic)")

	// Filter flags
	cmd.Flags().StringVarP(&convAgent, "agent", "a", "all", "Filter by agent: claude, codex, cursor, opencode, pi, all")
	cmd.Flags().StringVarP(&convProject, "project", "p", "", "Filter by project name or path")
	cmd.Flags().StringVar(&convSince, "since", "", "Conversations since: 1h, 7d, 2w, 1m, 1y")
	cmd.Flags().StringVar(&convAfter, "after", "", "Conversations after date (YYYY-MM-DD)")
	cmd.Flags().StringVar(&convBefore, "before", "", "Conversations before date (YYYY-MM-DD)")

	// Output flags
	cmd.Flags().BoolVar(&convJSON, "json", false, "Output as JSON")
	cmd.Flags().BoolVarP(&convQuiet, "quiet", "q", false, "Minimal output (session IDs only)")
	cmd.Flags().BoolVarP(&convVerbose, "verbose", "v", false, "Verbose output with full turn content")
	cmd.Flags().StringVar(&convFormat, "format", "default", "Output format: default, table, timeline")
	cmd.Flags().BoolVar(&convNoColor, "no-color", false, "Disable colored output")
	cmd.Flags().BoolVarP(&convInteractive, "interactive", "i", false, "Enter interactive mode after search")

	// Hybrid search weights
	cmd.Flags().Float64Var(&convSemanticWeight, "semantic-weight", 0.6, "Weight for semantic score in hybrid mode")
	cmd.Flags().Float64Var(&convBM25Weight, "bm25-weight", 0.4, "Weight for BM25 score in hybrid mode")
}

// Command implementations

func runConvSearch(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmd.Help()
	}
	query := args[0]

	ctx := context.Background()

	// Open conversation store
	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	// Create embedder
	embedder := embed.New()

	// Create searcher
	searcher := conv.NewSearcher(store, embedder)

	// Build search options
	opts := conv.DefaultSearchOptions()
	opts.Limit = convLimit
	opts.Threshold = convThreshold
	opts.Agent = conv.ParseAgentType(convAgent)
	opts.Project = convProject
	opts.UseHybrid = convHybrid
	opts.ExactMatch = convExact
	opts.SemanticWeight = convSemanticWeight
	opts.BM25Weight = convBM25Weight

	// Parse time filters
	if convSince != "" {
		duration, err := conv.ParseDuration(convSince)
		if err != nil {
			return fmt.Errorf("invalid --since: %w", err)
		}
		opts.Since = time.Now().Add(-duration)
	}
	if convAfter != "" {
		t, err := time.Parse("2006-01-02", convAfter)
		if err != nil {
			return fmt.Errorf("invalid --after date: %w", err)
		}
		opts.Since = t
	}
	if convBefore != "" {
		t, err := time.Parse("2006-01-02", convBefore)
		if err != nil {
			return fmt.Errorf("invalid --before date: %w", err)
		}
		opts.Before = t
	}

	// Search
	response, err := searcher.Search(ctx, query, opts)
	if err != nil {
		return err
	}

	// Output results
	return outputConvResults(response)
}

func runConvView(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	actions := conv.NewActions(store)
	session, err := actions.View(ctx, sessionID, conv.ViewOptions{
		Turn:    convTurn,
		JSONOut: convJSON,
	})
	if err != nil {
		return err
	}

	if convJSON {
		return json.NewEncoder(os.Stdout).Encode(session)
	}

	// Print formatted view
	fmt.Printf("Session: %s\n", session.ID)
	fmt.Printf("Agent:   %s\n", session.Agent)
	fmt.Printf("Project: %s\n", session.ProjectName)
	fmt.Printf("Date:    %s\n", session.StartedAt.Format("2006-01-02 15:04"))
	fmt.Printf("Turns:   %d\n", len(session.Turns))
	fmt.Println(strings.Repeat("─", 60))

	for i, turn := range session.Turns {
		fmt.Printf("\n[Turn %d]\n", i+1)
		fmt.Printf("USER:\n  %s\n", indent(turn.UserContent, "  "))
		fmt.Printf("\nASSISTANT:\n  %s\n", indent(turn.AssistContent, "  "))
		if i < len(session.Turns)-1 {
			fmt.Println(strings.Repeat("─", 40))
		}
	}

	return nil
}

func runConvResume(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	actions := conv.NewActions(store)
	result, err := actions.Resume(ctx, sessionID, conv.ResumeOptions{
		FromTurn: convResumeFrom,
		DryRun:   convResumeDryRun,
	})
	if err != nil {
		return err
	}

	if convResumeDryRun {
		fmt.Printf("Command: %s %s\n", result.Command, strings.Join(result.Args, " "))
		fmt.Printf("Project: %s\n", result.ProjectPath)
	} else if result.Executed {
		fmt.Println("Resuming conversation...")
	}

	return result.Error
}

func runConvContext(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	actions := conv.NewActions(store)
	result, err := actions.ExtractContext(ctx, sessionID, conv.ContextOptions{
		Turns:   convContextTurns,
		Summary: convContextSummary,
		Format:  convFormat,
		Copy:    convContextCopy,
	})
	if err != nil {
		return err
	}

	if convFormat == "json" {
		return json.NewEncoder(os.Stdout).Encode(result)
	}

	fmt.Print(result.FormattedOutput)
	if convContextCopy {
		fmt.Println("\n(Copied to clipboard)")
	}

	return nil
}

func runConvExport(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	actions := conv.NewActions(store)
	content, err := actions.Export(ctx, sessionID, conv.ExportOptions{
		Format: convFormat,
		Output: convOutput,
	})
	if err != nil {
		return err
	}

	if convOutput == "" {
		fmt.Print(content)
	} else {
		if err := os.WriteFile(convOutput, []byte(content), 0644); err != nil {
			return fmt.Errorf("failed to write file: %w", err)
		}
		fmt.Printf("Exported to %s\n", convOutput)
	}

	return nil
}

func runConvCopy(cmd *cobra.Command, args []string) error {
	sessionID := args[0]
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	actions := conv.NewActions(store)
	content, err := actions.Copy(ctx, sessionID, conv.CopyOptions{
		Turn:     convTurn,
		CodeOnly: convCopyCodeOnly,
		Full:     convCopyFull,
	})
	if err != nil {
		return err
	}

	// Show preview
	preview := content
	if len(preview) > 200 {
		preview = preview[:197] + "..."
	}
	fmt.Printf("Copied to clipboard:\n%s\n", preview)

	return nil
}

func runConvIndex(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	var cancel context.CancelFunc
	if convIndexWatch {
		ctx, cancel = context.WithCancel(ctx)
		signals := make(chan os.Signal, 1)
		signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
		go func() {
			<-signals
			cancel()
		}()
		defer func() {
			signal.Stop(signals)
			close(signals)
		}()
	}

	// Open or create store
	store, err := openConvStore()
	if err != nil {
		// Try to create new store
		cfg := conv.DefaultStoreConfig()
		store, err = conv.NewStore(cfg)
		if err != nil {
			return fmt.Errorf("failed to create store: %w", err)
		}
	}
	defer func() { _ = store.Close() }()

	// Create embedder
	embedder := embed.New()

	// Create indexer
	indexer := conv.NewIndexer(conv.IndexerConfig{
		Store:    store,
		Embedder: embedder,
		Force:    convIndexForce,
	})

	// Determine which parsers to use
	var parsers []parser.Parser
	if convIndexSource != "" {
		p, ok := parser.Get(conv.ParseAgentType(convIndexSource))
		if !ok {
			return fmt.Errorf("unknown source: %s", convIndexSource)
		}
		parsers = append(parsers, p)
	} else {
		parsers = parser.All()
	}

	if convIndexWatch {
		return watchConversations(ctx, indexer, parsers, convVerbose)
	}

	// Index from each parser
	var totalSessions, totalTurns int
	for _, p := range parsers {
		fmt.Printf("Indexing %s conversations...\n", p.AgentType())

		paths, err := p.Discover()
		if err != nil {
			fmt.Printf("  Warning: discovery failed: %v\n", err)
			continue
		}

		if len(paths) == 0 {
			fmt.Printf("  No conversations found\n")
			continue
		}

		for _, path := range paths {
			sessions, err := p.Parse(path)
			if err != nil {
				fmt.Printf("  Warning: parse failed for %s: %v\n", path, err)
				continue
			}

			if len(sessions) == 0 {
				continue
			}

			result, err := indexer.IndexSessions(ctx, sessions)
			if err != nil {
				fmt.Printf("  Warning: index failed: %v\n", err)
				continue
			}

			// Print errors from indexing
			for _, e := range result.Errors {
				fmt.Printf("  Warning: %v\n", e)
			}

			if result.SessionsIndexed > 0 {
				fmt.Printf("  Indexed %d sessions (%d turns) from %s\n",
					result.SessionsIndexed, result.TurnsIndexed, path)
			} else if result.SessionsFound > 0 {
				fmt.Printf("  Skipped %d sessions (already indexed) from %s\n",
					result.SessionsFound, path)
			}
			totalSessions += result.SessionsIndexed
			totalTurns += result.TurnsIndexed
		}
	}

	fmt.Printf("\nTotal: %d sessions, %d turns indexed\n", totalSessions, totalTurns)
	return nil
}

func runConvStatus(cmd *cobra.Command, args []string) error {
	ctx := context.Background()

	store, err := openConvStore()
	if err != nil {
		fmt.Println("No conversation index found.")
		fmt.Println("Run 'sgrep conv index' to create one.")
		return nil
	}
	defer func() { _ = store.Close() }()

	stats, err := store.GetStats(ctx)
	if err != nil {
		return err
	}

	if convJSON {
		return json.NewEncoder(os.Stdout).Encode(stats)
	}

	fmt.Println("Conversation Index Status")
	fmt.Println(strings.Repeat("─", 40))
	fmt.Printf("Sessions:    %d\n", stats.TotalSessions)
	fmt.Printf("Turns:       %d\n", stats.TotalTurns)
	fmt.Printf("Size:        %s\n", formatBytes(stats.IndexSizeBytes))
	fmt.Printf("Last Indexed: %s\n", stats.LastIndexed.Format("2006-01-02 15:04:05"))
	fmt.Println()
	fmt.Println("Sessions by Agent:")
	for agent, count := range stats.SessionsByAgent {
		fmt.Printf("  %s: %d\n", agent, count)
	}

	return nil
}

// Helper functions

func openConvStore() (*conv.Store, error) {
	cfg := conv.DefaultStoreConfig()
	return conv.NewStore(cfg)
}

func outputConvResults(response *conv.SearchResponse) error {
	if convJSON {
		return json.NewEncoder(os.Stdout).Encode(response)
	}

	if len(response.Results) == 0 {
		if !convQuiet {
			fmt.Println("No conversations found")
		}
		return nil
	}

	if convQuiet {
		for _, hit := range response.Results {
			fmt.Println(hit.SessionID)
		}
		return nil
	}

	fmt.Printf("Found %d conversations matching %q\n\n", response.TotalHits, response.Query)

	for i, hit := range response.Results {
		if convVerbose {
			printVerboseHit(i+1, hit)
		} else {
			printDefaultHit(i+1, hit)
		}
	}

	if !convNoColor {
		fmt.Println(strings.Repeat("─", 60))
		fmt.Println("Actions: [v]iew  [r]esume  [c]opy  [e]xport  [q]uit")
	}

	return nil
}

func printDefaultHit(index int, hit conv.ConversationHit) {
	fmt.Printf("[%d] %s  %s  %s\n",
		index, hit.Agent, hit.ProjectName, hit.RelativeTime)
	fmt.Printf("    ID: %s\n", hit.SessionID)
	fmt.Printf("    YOU: %q\n", hit.MatchedTurn.UserSnip)
	fmt.Printf("    %s: %q\n", strings.ToUpper(string(hit.Agent)), hit.MatchedTurn.AssistSnip)
	fmt.Printf("    └─ %d turns · score: %.2f\n\n", hit.TotalTurns, hit.Score)
}

func printVerboseHit(index int, hit conv.ConversationHit) {
	fmt.Println(strings.Repeat("━", 60))
	fmt.Printf("[%d] %s · %s\n", index, hit.Agent, hit.ProjectName)
	fmt.Println(strings.Repeat("━", 60))
	fmt.Printf("Session:  %s\n", hit.SessionID)
	fmt.Printf("Started:  %s (%s)\n", hit.StartedAt.Format("2006-01-02 15:04"), hit.RelativeTime)
	fmt.Printf("Project:  %s\n", hit.ProjectPath)
	fmt.Printf("Turns:    %d · Score: %.2f\n\n", hit.TotalTurns, hit.Score)

	fmt.Printf("[Turn %d/%d]\n", hit.MatchedTurn.TurnIndex+1, hit.TotalTurns)
	fmt.Println(strings.Repeat("─", 40))
	fmt.Printf("USER:\n  %s\n\n", indent(hit.MatchedTurn.FullUser, "  "))
	fmt.Printf("%s:\n  %s\n\n", strings.ToUpper(string(hit.Agent)), indent(hit.MatchedTurn.FullAssist, "  "))

	fmt.Printf("Resume: %s\n", hit.Actions.Resume)
	fmt.Println()
}

func indent(s string, prefix string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		if i > 0 {
			lines[i] = prefix + line
		}
	}
	return strings.Join(lines, "\n")
}
