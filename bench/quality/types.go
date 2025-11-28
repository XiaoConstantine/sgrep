// Package quality provides IR-style evaluation for semantic code search.
package quality

import (
	"encoding/json"
	"os"
)

// Relevance represents graded relevance judgments for search results.
type Relevance int

const (
	RelNone          Relevance = 0 // Not relevant
	RelRelevant      Relevance = 1 // Relevant (secondary/supporting)
	RelHighlyRelevant Relevance = 2 // Highly relevant (primary target)
)

// LabeledResult represents a file with its relevance judgment.
type LabeledResult struct {
	File string    `json:"file"` // Repo-relative path
	Rel  Relevance `json:"rel"`  // Relevance grade (0/1/2)
}

// QueryCase represents a single evaluation query with ground truth.
type QueryCase struct {
	Query        string          `json:"query"`
	Judgments    []LabeledResult `json:"judgments"`
	GrepPatterns []string        `json:"grep_patterns,omitempty"`
	Category     string          `json:"category,omitempty"` // e.g., "conceptual", "api", "edge_case"
}

// Dataset represents a complete evaluation dataset.
type Dataset struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Corpus      string      `json:"corpus"`      // Path or identifier of the codebase
	CorpusHash  string      `json:"corpus_hash"` // Git commit hash for reproducibility
	Queries     []QueryCase `json:"queries"`
}

// SearchResult represents a single result from a search tool.
type SearchResult struct {
	File      string  `json:"file"`
	Score     float64 `json:"score,omitempty"`
	StartLine int     `json:"start_line,omitempty"`
	EndLine   int     `json:"end_line,omitempty"`
}

// EvalResult holds evaluation metrics for a single query.
type EvalResult struct {
	Query       string  `json:"query"`
	Tool        string  `json:"tool"` // "sgrep", "ripgrep", "osgrep"
	MRR         float64 `json:"mrr"`
	NDCG5       float64 `json:"ndcg@5"`
	NDCG10      float64 `json:"ndcg@10"`
	MAP         float64 `json:"map"`
	PrecisionAt5  float64 `json:"p@5"`
	PrecisionAt10 float64 `json:"p@10"`
	RecallAt5     float64 `json:"r@5"`
	RecallAt10    float64 `json:"r@10"`
	LatencyMs   float64 `json:"latency_ms"`
	TokensUsed  int     `json:"tokens_used,omitempty"`
}

// Summary holds aggregated metrics across all queries.
type Summary struct {
	Tool          string  `json:"tool"`
	NumQueries    int     `json:"num_queries"`
	MeanMRR       float64 `json:"mean_mrr"`
	MeanNDCG5     float64 `json:"mean_ndcg@5"`
	MeanNDCG10    float64 `json:"mean_ndcg@10"`
	MeanMAP       float64 `json:"mean_map"`
	MeanP5        float64 `json:"mean_p@5"`
	MeanP10       float64 `json:"mean_p@10"`
	MeanR5        float64 `json:"mean_r@5"`
	MeanR10       float64 `json:"mean_r@10"`
	MeanLatencyMs float64 `json:"mean_latency_ms"`
	TotalTokens   int     `json:"total_tokens,omitempty"`
}

// Report holds the complete evaluation report.
type Report struct {
	Dataset   string       `json:"dataset"`
	Timestamp string       `json:"timestamp"`
	Tools     []string     `json:"tools"`
	Results   []EvalResult `json:"results"`
	Summaries []Summary    `json:"summaries"`
}

// LoadDataset loads an evaluation dataset from a JSON file.
func LoadDataset(path string) (*Dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var ds Dataset
	if err := json.Unmarshal(data, &ds); err != nil {
		return nil, err
	}

	return &ds, nil
}

// SaveReport saves an evaluation report to a JSON file.
func SaveReport(path string, report *Report) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// RelevanceMap builds a map from file path to relevance for efficient lookup.
func (q *QueryCase) RelevanceMap() map[string]Relevance {
	m := make(map[string]Relevance, len(q.Judgments))
	for _, j := range q.Judgments {
		m[j.File] = j.Rel
	}
	return m
}

// TotalRelevant returns the count of relevant items (Rel > 0).
func (q *QueryCase) TotalRelevant() int {
	count := 0
	for _, j := range q.Judgments {
		if j.Rel > RelNone {
			count++
		}
	}
	return count
}

// ToRelevances converts search results to a relevance slice using ground truth.
func ToRelevances(results []SearchResult, relMap map[string]Relevance) []Relevance {
	rels := make([]Relevance, len(results))
	for i, r := range results {
		rels[i] = relMap[r.File] // defaults to RelNone if not found
	}
	return rels
}
