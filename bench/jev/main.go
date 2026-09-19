// Command jev-experiment runs an explicit-opt-in hosted Jev reranking experiment.
// It is benchmark-only and does not alter sgrep's production --rerank behavior.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/embed"
	"github.com/XiaoConstantine/sgrep/pkg/rerank"
	"github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/store"
)

type resultRecord struct {
	File      string  `json:"file"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float64 `json:"score"`
}

type queryRecord struct {
	Query            string         `json:"query"`
	ExpectedAnswer   bool           `json:"expected_answer"`
	Metrics          queryMetrics   `json:"metrics"`
	LatencyMS        float64        `json:"latency_ms"`
	Results          []resultRecord `json:"results"`
	Jev              *jevRunStats   `json:"jev,omitempty"`
	PredictedAny     *bool          `json:"predicted_any_answer,omitempty"`
	FallbackVerified bool           `json:"fallback_verified,omitempty"`
}

type armSummary struct {
	Name             string        `json:"name"`
	Queries          int           `json:"queries"`
	AnsweredQueries  int           `json:"answered_queries"`
	MRR              float64       `json:"mrr"`
	NDCG10           float64       `json:"ndcg@10"`
	Recall10         float64       `json:"recall@10"`
	PromotionErrors  int           `json:"implementation_vs_prose_promotion_errors"`
	P50LatencyMS     float64       `json:"p50_end_to_end_latency_ms"`
	P95LatencyMS     float64       `json:"p95_end_to_end_latency_ms"`
	InputTokens      int           `json:"jev_input_tokens,omitempty"`
	OutputTokens     int           `json:"jev_output_tokens,omitempty"`
	CostUSD          float64       `json:"jev_cost_usd,omitempty"`
	Failures         int           `json:"failures"`
	Fallbacks        int           `json:"fallbacks"`
	AnswerabilityHit int           `json:"answerability_correct,omitempty"`
	Records          []queryRecord `json:"records"`
}

type artifact struct {
	Experiment       string       `json:"experiment"`
	CreatedAt        string       `json:"created_at"`
	SourceCommit     string       `json:"source_commit"`
	CorpusCommit     string       `json:"corpus_commit"`
	Dataset          string       `json:"dataset"`
	Split            string       `json:"split"`
	Model            string       `json:"jev_model"`
	Shortlist        int          `json:"jev_shortlist"`
	ScoreScale       float64      `json:"jev_score_scale"`
	PricePerMTok     float64      `json:"jev_price_per_million_input_tokens"`
	Privacy          string       `json:"privacy"`
	Runtime          string       `json:"runtime"`
	LiveJev          bool         `json:"live_jev"`
	BlockedReason    string       `json:"blocked_reason,omitempty"`
	FallbackVerified bool         `json:"failure_fallback_verified"`
	Arms             []armSummary `json:"arms"`
}

type arm struct {
	name        string
	hybrid      bool
	colbert     bool
	reranker    *rerank.Reranker
	jev         *jevProxy
	available   bool
	unavailable string
}

func main() {
	var corpusPath, indexPath, datasetPath, outputPath, split string
	var localRerankerEndpoint, localRerankerName string
	var scoreScale float64
	var allowUpload, localOnly, skipJina, verifyFallback bool
	flag.StringVar(&corpusPath, "corpus", "", "path to the pinned dspy-go checkout")
	flag.StringVar(&indexPath, "index", "", "path to sgrep index.db (default: derive from corpus and SGREP_HOME)")
	flag.StringVar(&datasetPath, "dataset", "bench/jev/dataset.json", "fixed evaluation dataset")
	flag.StringVar(&outputPath, "output", "bench/jev/results.json", "JSON artifact path")
	flag.StringVar(&split, "split", "heldout", "dataset split to evaluate")
	flag.Float64Var(&scoreScale, "jev-score-scale", 1, "shrink Jev score differences toward neutral (benchmark sensitivity only; 0..1)")
	flag.BoolVar(&allowUpload, "allow-source-upload", false, "explicitly allow sending queries and shortlisted source chunks to TypeSafe")
	flag.BoolVar(&localOnly, "local-only", false, "run local comparison arms without calling Jev")
	flag.BoolVar(&skipJina, "skip-jina", false, "skip the local Jina comparison if its model is unavailable")
	flag.BoolVar(&verifyFallback, "verify-fallback", true, "inject one Jev failure and verify exact baseline fallback")
	flag.StringVar(&localRerankerEndpoint, "local-reranker-endpoint", "", "benchmark an additional llama.cpp-compatible reranker endpoint")
	flag.StringVar(&localRerankerName, "local-reranker-name", "local reranker", "artifact label for --local-reranker-endpoint")
	flag.Parse()

	if err := run(corpusPath, indexPath, datasetPath, outputPath, split, localRerankerEndpoint, localRerankerName, scoreScale, allowUpload, localOnly, skipJina, verifyFallback); err != nil {
		fmt.Fprintln(os.Stderr, "jev experiment:", err)
		os.Exit(1)
	}
}

func run(corpusPath, indexPath, datasetPath, outputPath, split, localRerankerEndpoint, localRerankerName string, scoreScale float64, allowUpload, localOnly, skipJina, verifyFallback bool) error {
	if scoreScale < 0 || scoreScale > 1 {
		return errors.New("--jev-score-scale must be between 0 and 1")
	}
	if !localOnly && !allowUpload {
		return errors.New("refusing hosted calls without --allow-source-upload; TypeSafe receives each query and up to 10 shortlisted source chunks")
	}
	apiKey := strings.TrimSpace(os.Getenv("TYPESAFE_API_KEY"))
	if !localOnly && apiKey == "" {
		return errors.New("TYPESAFE_API_KEY is unavailable")
	}
	if corpusPath == "" {
		return errors.New("--corpus is required")
	}
	corpusPath, err := filepath.Abs(corpusPath)
	if err != nil {
		return err
	}

	ds, err := loadDataset(datasetPath)
	if err != nil {
		return err
	}
	corpusCommit, err := gitCommit(corpusPath)
	if err != nil {
		return err
	}
	if corpusCommit != ds.CorpusHash {
		return fmt.Errorf("corpus commit %s does not match dataset pin %s", corpusCommit, ds.CorpusHash)
	}
	if indexPath == "" {
		indexPath, err = derivedIndexPath(corpusPath)
		if err != nil {
			return err
		}
	}

	if localOnly {
		fmt.Fprintln(os.Stderr, "LOCAL-ONLY MODE: no query or source content will be sent to TypeSafe; Jev quality conclusions remain blocked.")
	} else {
		fmt.Fprintln(os.Stderr, "PRIVACY NOTICE: explicit opt-in accepted; queries and shortlisted source content will be sent to TypeSafe's hosted API.")
		fmt.Fprintf(os.Stderr, "Pinned model %s; shortlist %d; no request bodies will be written to the artifact.\n", jevModel, jevShortlistLimit)
	}

	s, err := store.OpenForSearch(indexPath)
	if err != nil {
		return fmt.Errorf("open search index: %w", err)
	}
	defer func() { _ = s.Close() }()
	if err := store.EnsureFTS5IfNeeded(s); err != nil {
		return fmt.Errorf("ensure FTS5: %w", err)
	}

	indexDir := filepath.Dir(indexPath)
	segments, err := store.OpenMMapSegmentStore(indexDir, 768)
	if err != nil {
		return fmt.Errorf("open ColBERT segments: %w", err)
	}
	defer func() { _ = segments.Close() }()

	proxy := newJevProxy(apiKey, scoreScale)
	defer proxy.close()
	jevReranker := rerank.NewWithConfig(rerank.Config{
		Endpoint:  proxy.endpoint(),
		Timeout:   60 * time.Second,
		AutoStart: false,
	})

	arms := []arm{
		{name: "balanced baseline", hybrid: true, available: true},
		{name: "balanced + existing Jina", hybrid: true},
		{name: "quality / ColBERT", hybrid: true, colbert: true, available: true},
	}
	if !localOnly {
		arms = append(arms,
			arm{name: "balanced + Jev", hybrid: true, reranker: jevReranker, jev: proxy, available: true},
			arm{name: "ColBERT shortlist + Jev", hybrid: true, colbert: true, reranker: jevReranker, jev: proxy, available: true},
		)
	}
	if localRerankerEndpoint != "" {
		localReranker := rerank.NewWithConfig(rerank.Config{
			Endpoint:  strings.TrimRight(localRerankerEndpoint, "/"),
			Timeout:   5 * time.Minute,
			AutoStart: false,
		})
		arms = append(arms,
			arm{name: "balanced + " + localRerankerName, hybrid: true, reranker: localReranker, available: true},
			arm{name: "ColBERT shortlist + " + localRerankerName, hybrid: true, colbert: true, reranker: localReranker, available: true},
		)
	}
	if rerank.RerankerAvailable() {
		jina, err := rerank.New()
		if err != nil {
			return fmt.Errorf("initialize local Jina: %w", err)
		}
		arms[1].reranker = jina
		arms[1].available = true
	} else if skipJina {
		arms[1].unavailable = "local Jina model unavailable; explicitly skipped"
	} else {
		return errors.New("local Jina model unavailable; run `sgrep setup --with-rerank` or pass --skip-jina")
	}

	var cases []queryCase
	for _, item := range ds.Cases {
		if item.Split == split {
			cases = append(cases, item)
		}
	}
	if len(cases) == 0 {
		return fmt.Errorf("dataset has no %q cases", split)
	}

	report := artifact{
		Experiment:   "isolated Jev code-search reranking",
		CreatedAt:    time.Now().UTC().Format(time.RFC3339),
		SourceCommit: sourceCommit(),
		CorpusCommit: corpusCommit,
		Dataset:      ds.Name,
		Split:        split,
		Model:        jevModel,
		Shortlist:    jevShortlistLimit,
		ScoreScale:   scoreScale,
		PricePerMTok: jevPricePerMTok,
		Privacy:      "Explicit opt-in: query and shortlisted source content transmitted to TypeSafe. Artifact excludes source and API credentials.",
		Runtime:      runtime.Version(),
		LiveJev:      !localOnly,
	}
	if localOnly {
		report.Privacy = "Local-only run; no query or source content transmitted to TypeSafe."
		report.BlockedReason = "local alternatives run; no live Jev calls"
	}

	baselineByQuery := make(map[string][]search.Result)
	for _, candidateArm := range arms {
		if !candidateArm.available {
			fmt.Fprintf(os.Stderr, "Skipping %s: %s\n", candidateArm.name, candidateArm.unavailable)
			continue
		}
		fmt.Fprintf(os.Stderr, "Running %s (%d held-out cases)...\n", candidateArm.name, len(cases))
		armEmbedder := embed.New()
		if _, err := armEmbedder.EmbedQuery(context.Background(), "jev experiment warmup"); err != nil {
			return fmt.Errorf("warm %s embedder: %w", candidateArm.name, err)
		}
		summary, err := runArm(context.Background(), s, segments, armEmbedder, candidateArm, cases)
		if err != nil {
			return fmt.Errorf("%s: %w", candidateArm.name, err)
		}
		if candidateArm.name == "balanced baseline" {
			for _, record := range summary.Records {
				baselineByQuery[record.Query] = recordsToResults(record.Results)
			}
		}
		report.Arms = append(report.Arms, summary)
	}

	if verifyFallback {
		first := cases[0]
		proxy.failOnce()
		fallbackArm := arm{name: "Jev failure fallback", hybrid: true, reranker: jevReranker, jev: proxy, available: true}
		fallbackEmbedder := embed.New()
		if _, err := fallbackEmbedder.EmbedQuery(context.Background(), "jev experiment warmup"); err != nil {
			return fmt.Errorf("warm fallback embedder: %w", err)
		}
		summary, err := runArm(context.Background(), s, segments, fallbackEmbedder, fallbackArm, []queryCase{first})
		if err != nil {
			return fmt.Errorf("fallback check: %w", err)
		}
		got := recordsToResults(summary.Records[0].Results)
		want := baselineByQuery[first.Query]
		if !sameResults(got, want) {
			return errors.New("injected Jev failure did not preserve exact balanced baseline results")
		}
		report.FallbackVerified = true
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return err
	}
	if err := os.WriteFile(outputPath, append(data, '\n'), 0644); err != nil {
		return err
	}
	printSummary(report)
	return nil
}

func runArm(ctx context.Context, s store.Storer, segments store.ColBERTSegmentStorer, embedder *embed.Embedder, candidateArm arm, cases []queryCase) (armSummary, error) {
	searcher := search.NewWithConfig(search.Config{
		Store:         s,
		SegmentStore:  segments,
		Embedder:      embedder,
		Reranker:      candidateArm.reranker,
		ColBERTScorer: nil,
		CacheSize:     len(cases) * 2,
	})
	var fallbackSearcher *search.Searcher
	if candidateArm.jev != nil {
		fallbackSearcher = search.NewWithConfig(search.Config{
			Store:         s,
			SegmentStore:  segments,
			Embedder:      embedder,
			ColBERTScorer: nil,
			CacheSize:     len(cases) * 2,
		})
	}
	summary := armSummary{Name: candidateArm.name, Queries: len(cases)}
	var durations []time.Duration
	for _, item := range cases {
		opts := search.DefaultSearchOptions()
		opts.Limit = 10
		opts.Threshold = 1.5
		opts.IncludeTests = false
		opts.Deduplicate = true
		opts.UseHybrid = candidateArm.hybrid
		opts.UseColBERT = candidateArm.colbert
		opts.UseRerank = candidateArm.reranker != nil
		opts.RerankTopK = 50

		started := time.Now()
		results, err := searcher.SearchWithOptions(ctx, item.Query, opts)
		if err != nil {
			return armSummary{}, err
		}
		var stats jevRunStats
		if candidateArm.jev != nil {
			stats = candidateArm.jev.takeStats(item.Query)
			if stats.Fallback {
				fallbackOpts := opts
				fallbackOpts.UseRerank = false
				results, err = fallbackSearcher.SearchWithOptions(ctx, item.Query, fallbackOpts)
				if err != nil {
					return armSummary{}, fmt.Errorf("fail-open search: %w", err)
				}
			}
		}
		elapsed := time.Since(started)
		durations = append(durations, elapsed)
		metrics := measure(results, item)
		record := queryRecord{
			Query:          item.Query,
			ExpectedAnswer: item.AnswersQuery,
			Metrics:        metrics,
			LatencyMS:      float64(elapsed.Microseconds()) / 1000,
			Results:        resultRecords(results),
		}
		if item.AnswersQuery {
			summary.AnsweredQueries++
			summary.MRR += metrics.MRR
			summary.NDCG10 += metrics.NDCG10
			summary.Recall10 += metrics.Recall10
			summary.PromotionErrors += metrics.PromotionErrors
		}
		if candidateArm.jev != nil {
			record.Jev = &stats
			predicted := stats.AnyAnswerScore >= 0.5
			record.PredictedAny = &predicted
			if predicted == item.AnswersQuery {
				summary.AnswerabilityHit++
			}
			summary.InputTokens += stats.InputTokens
			summary.OutputTokens += stats.OutputTokens
			if stats.Failure != "" {
				summary.Failures++
			}
			if stats.Fallback {
				summary.Fallbacks++
			}
		}
		summary.Records = append(summary.Records, record)
	}
	if summary.AnsweredQueries > 0 {
		n := float64(summary.AnsweredQueries)
		summary.MRR /= n
		summary.NDCG10 /= n
		summary.Recall10 /= n
	}
	summary.P50LatencyMS = percentile(durations, 0.50)
	summary.P95LatencyMS = percentile(durations, 0.95)
	summary.CostUSD = float64(summary.InputTokens) / 1_000_000 * jevPricePerMTok
	return summary, nil
}

func loadDataset(path string) (dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return dataset{}, err
	}
	var result dataset
	if err := json.Unmarshal(data, &result); err != nil {
		return dataset{}, err
	}
	return result, nil
}

func derivedIndexPath(corpusPath string) (string, error) {
	home := strings.TrimSpace(os.Getenv("SGREP_HOME"))
	if home == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		home = filepath.Join(userHome, ".sgrep")
	}
	resolved, err := filepath.EvalSymlinks(corpusPath)
	if err == nil {
		corpusPath = resolved
	}
	hash := sha256.Sum256([]byte(filepath.Clean(corpusPath)))
	path := filepath.Join(home, "repos", fmt.Sprintf("%x", hash[:6]), "index.db")
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("derived index %s is unavailable: %w", path, err)
	}
	return path, nil
}

func gitCommit(dir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(dir, ".git", "HEAD"))
	if err != nil {
		return "", fmt.Errorf("read corpus git HEAD: %w", err)
	}
	head := strings.TrimSpace(string(data))
	if !strings.HasPrefix(head, "ref: ") {
		return head, nil
	}
	ref := strings.TrimPrefix(head, "ref: ")
	data, err = os.ReadFile(filepath.Join(dir, ".git", filepath.FromSlash(ref)))
	if err != nil {
		return "", fmt.Errorf("read corpus git ref: %w", err)
	}
	return strings.TrimSpace(string(data)), nil
}

func sourceCommit() string {
	data, err := os.ReadFile(filepath.Join(".git", "HEAD"))
	if err != nil {
		return "unknown"
	}
	head := strings.TrimSpace(string(data))
	if !strings.HasPrefix(head, "ref: ") {
		return head
	}
	data, err = os.ReadFile(filepath.Join(".git", filepath.FromSlash(strings.TrimPrefix(head, "ref: "))))
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(data))
}

func resultRecords(results []search.Result) []resultRecord {
	records := make([]resultRecord, len(results))
	for i, result := range results {
		records[i] = resultRecord{result.FilePath, result.StartLine, result.EndLine, result.Score}
	}
	return records
}

func recordsToResults(records []resultRecord) []search.Result {
	results := make([]search.Result, len(records))
	for i, record := range records {
		results[i] = search.Result{FilePath: record.File, StartLine: record.StartLine, EndLine: record.EndLine, Score: record.Score}
	}
	return results
}

func sameResults(a, b []search.Result) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].FilePath != b[i].FilePath || a[i].StartLine != b[i].StartLine || a[i].EndLine != b[i].EndLine || a[i].Score != b[i].Score {
			return false
		}
	}
	return true
}

func printSummary(report artifact) {
	fmt.Println("arm\tMRR\tNDCG@10\tR@10\tpromotion-errors\tp50-ms\tp95-ms\tJev-input-tokens\tcost-usd\tfailures")
	for _, arm := range report.Arms {
		fmt.Printf("%s\t%.3f\t%.3f\t%.3f\t%d\t%.1f\t%.1f\t%d\t%.6f\t%d\n",
			arm.Name, arm.MRR, arm.NDCG10, arm.Recall10, arm.PromotionErrors,
			arm.P50LatencyMS, arm.P95LatencyMS, arm.InputTokens, arm.CostUSD, arm.Failures)
	}
	fmt.Printf("failure fallback verified: %v\n", report.FallbackVerified)
}
