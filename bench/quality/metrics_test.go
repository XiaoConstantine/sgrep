package quality

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestMRR(t *testing.T) {
	tests := []struct {
		name     string
		rels     []Relevance
		expected float64
	}{
		{
			name:     "first is relevant",
			rels:     []Relevance{RelRelevant, RelNone, RelNone},
			expected: 1.0,
		},
		{
			name:     "second is relevant",
			rels:     []Relevance{RelNone, RelRelevant, RelNone},
			expected: 0.5,
		},
		{
			name:     "third is relevant",
			rels:     []Relevance{RelNone, RelNone, RelRelevant},
			expected: 1.0 / 3.0,
		},
		{
			name:     "highly relevant counts",
			rels:     []Relevance{RelNone, RelHighlyRelevant, RelNone},
			expected: 0.5,
		},
		{
			name:     "no relevant",
			rels:     []Relevance{RelNone, RelNone, RelNone},
			expected: 0,
		},
		{
			name:     "empty",
			rels:     []Relevance{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MRR(tt.rels)
			if !floatEquals(got, tt.expected) {
				t.Errorf("MRR() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPrecisionAtK(t *testing.T) {
	tests := []struct {
		name     string
		rels     []Relevance
		k        int
		expected float64
	}{
		{
			name:     "all relevant at k=3",
			rels:     []Relevance{RelRelevant, RelRelevant, RelRelevant, RelNone},
			k:        3,
			expected: 1.0,
		},
		{
			name:     "half relevant at k=4",
			rels:     []Relevance{RelRelevant, RelNone, RelRelevant, RelNone},
			k:        4,
			expected: 0.5,
		},
		{
			name:     "none relevant",
			rels:     []Relevance{RelNone, RelNone, RelNone},
			k:        3,
			expected: 0,
		},
		{
			name:     "k larger than results",
			rels:     []Relevance{RelRelevant, RelRelevant},
			k:        5,
			expected: 1.0,
		},
		{
			name:     "k=0",
			rels:     []Relevance{RelRelevant},
			k:        0,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PrecisionAtK(tt.rels, tt.k)
			if !floatEquals(got, tt.expected) {
				t.Errorf("PrecisionAtK() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestRecallAtK(t *testing.T) {
	tests := []struct {
		name          string
		rels          []Relevance
		totalRelevant int
		k             int
		expected      float64
	}{
		{
			name:          "all found at k=3",
			rels:          []Relevance{RelRelevant, RelRelevant, RelRelevant},
			totalRelevant: 3,
			k:             3,
			expected:      1.0,
		},
		{
			name:          "half found at k=2",
			rels:          []Relevance{RelRelevant, RelRelevant, RelNone, RelNone},
			totalRelevant: 4,
			k:             2,
			expected:      0.5,
		},
		{
			name:          "none found",
			rels:          []Relevance{RelNone, RelNone},
			totalRelevant: 2,
			k:             2,
			expected:      0,
		},
		{
			name:          "zero total relevant",
			rels:          []Relevance{RelRelevant},
			totalRelevant: 0,
			k:             1,
			expected:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RecallAtK(tt.rels, tt.totalRelevant, tt.k)
			if !floatEquals(got, tt.expected) {
				t.Errorf("RecallAtK() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestAveragePrecision(t *testing.T) {
	tests := []struct {
		name          string
		rels          []Relevance
		totalRelevant int
		expected      float64
	}{
		{
			name:          "perfect ranking",
			rels:          []Relevance{RelRelevant, RelRelevant, RelNone, RelNone},
			totalRelevant: 2,
			expected:      1.0, // (1/1 + 2/2) / 2 = 1.0
		},
		{
			name:          "interleaved",
			rels:          []Relevance{RelRelevant, RelNone, RelRelevant, RelNone},
			totalRelevant: 2,
			expected:      (1.0 + 2.0/3.0) / 2.0, // = 0.833...
		},
		{
			name:          "relevant at end",
			rels:          []Relevance{RelNone, RelNone, RelRelevant},
			totalRelevant: 1,
			expected:      1.0 / 3.0,
		},
		{
			name:          "no relevant found",
			rels:          []Relevance{RelNone, RelNone, RelNone},
			totalRelevant: 2,
			expected:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AveragePrecision(tt.rels, tt.totalRelevant)
			if !floatEquals(got, tt.expected) {
				t.Errorf("AveragePrecision() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestDCG(t *testing.T) {
	tests := []struct {
		name     string
		rels     []Relevance
		k        int
		expected float64
	}{
		{
			name:     "single highly relevant at pos 1",
			rels:     []Relevance{RelHighlyRelevant},
			k:        1,
			expected: 3.0 / math.Log2(2), // (2^2 - 1) / log2(2) = 3 / 1 = 3.0
		},
		{
			name:     "binary relevance",
			rels:     []Relevance{RelRelevant, RelRelevant},
			k:        2,
			expected: (1.0 / math.Log2(2)) + (1.0 / math.Log2(3)), // 1.0 + 0.63...
		},
		{
			name:     "no relevant",
			rels:     []Relevance{RelNone, RelNone},
			k:        2,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DCG(tt.rels, tt.k)
			if !floatEquals(got, tt.expected) {
				t.Errorf("DCG() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestNDCG(t *testing.T) {
	tests := []struct {
		name     string
		rels     []Relevance
		k        int
		expected float64
	}{
		{
			name:     "perfect order",
			rels:     []Relevance{RelHighlyRelevant, RelRelevant, RelNone},
			k:        3,
			expected: 1.0,
		},
		{
			name:     "reversed order",
			rels:     []Relevance{RelNone, RelRelevant, RelHighlyRelevant},
			k:        3,
			expected: 0.0, // Will be less than 1
		},
		{
			name:     "all same relevance",
			rels:     []Relevance{RelRelevant, RelRelevant, RelRelevant},
			k:        3,
			expected: 1.0,
		},
		{
			name:     "empty",
			rels:     []Relevance{},
			k:        3,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NDCG(tt.rels, tt.k)
			// For "reversed order" test, just check it's less than 1
			if tt.name == "reversed order" {
				if got >= 1.0 || got < 0 {
					t.Errorf("NDCG() = %v, want value in (0, 1)", got)
				}
				return
			}
			if !floatEquals(got, tt.expected) {
				t.Errorf("NDCG() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestF1(t *testing.T) {
	tests := []struct {
		name      string
		precision float64
		recall    float64
		expected  float64
	}{
		{
			name:      "perfect",
			precision: 1.0,
			recall:    1.0,
			expected:  1.0,
		},
		{
			name:      "balanced",
			precision: 0.5,
			recall:    0.5,
			expected:  0.5,
		},
		{
			name:      "zero precision",
			precision: 0,
			recall:    1.0,
			expected:  0,
		},
		{
			name:      "zero both",
			precision: 0,
			recall:    0,
			expected:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := F1(tt.precision, tt.recall)
			if !floatEquals(got, tt.expected) {
				t.Errorf("F1() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestJaccardSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		set1     []string
		set2     []string
		expected float64
	}{
		{
			name:     "identical",
			set1:     []string{"a", "b", "c"},
			set2:     []string{"a", "b", "c"},
			expected: 1.0,
		},
		{
			name:     "half overlap",
			set1:     []string{"a", "b"},
			set2:     []string{"b", "c"},
			expected: 1.0 / 3.0, // intersection=1, union=3
		},
		{
			name:     "no overlap",
			set1:     []string{"a", "b"},
			set2:     []string{"c", "d"},
			expected: 0,
		},
		{
			name:     "both empty",
			set1:     []string{},
			set2:     []string{},
			expected: 1.0,
		},
		{
			name:     "one empty",
			set1:     []string{"a"},
			set2:     []string{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := JaccardSimilarity(tt.set1, tt.set2)
			if !floatEquals(got, tt.expected) {
				t.Errorf("JaccardSimilarity() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestDiversity(t *testing.T) {
	tests := []struct {
		name     string
		results  []SearchResult
		expected float64
	}{
		{
			name: "all unique",
			results: []SearchResult{
				{File: "a.go"},
				{File: "b.go"},
				{File: "c.go"},
			},
			expected: 1.0,
		},
		{
			name: "all same",
			results: []SearchResult{
				{File: "a.go"},
				{File: "a.go"},
				{File: "a.go"},
			},
			expected: 1.0 / 3.0,
		},
		{
			name: "half duplicates",
			results: []SearchResult{
				{File: "a.go"},
				{File: "a.go"},
				{File: "b.go"},
				{File: "b.go"},
			},
			expected: 0.5,
		},
		{
			name:     "empty",
			results:  []SearchResult{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Diversity(tt.results)
			if !floatEquals(got, tt.expected) {
				t.Errorf("Diversity() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestComputeAllMetrics(t *testing.T) {
	query := &QueryCase{
		Query: "test query",
		Judgments: []LabeledResult{
			{File: "a.go", Rel: RelHighlyRelevant},
			{File: "b.go", Rel: RelRelevant},
			{File: "c.go", Rel: RelRelevant},
		},
	}

	results := []SearchResult{
		{File: "a.go"},
		{File: "x.go"}, // not relevant
		{File: "b.go"},
		{File: "y.go"}, // not relevant
		{File: "c.go"},
	}

	eval := ComputeAllMetrics(results, query)

	if eval.Query != "test query" {
		t.Errorf("Query = %v, want 'test query'", eval.Query)
	}

	// MRR should be 1.0 (first result is relevant)
	if !floatEquals(eval.MRR, 1.0) {
		t.Errorf("MRR = %v, want 1.0", eval.MRR)
	}

	// P@5 should be 3/5 = 0.6
	if !floatEquals(eval.PrecisionAt5, 0.6) {
		t.Errorf("P@5 = %v, want 0.6", eval.PrecisionAt5)
	}

	// R@5 should be 3/3 = 1.0
	if !floatEquals(eval.RecallAt5, 1.0) {
		t.Errorf("R@5 = %v, want 1.0", eval.RecallAt5)
	}
}

func TestAggregateSummary(t *testing.T) {
	results := []EvalResult{
		{MRR: 1.0, NDCG5: 0.8, MAP: 0.9, LatencyMs: 100},
		{MRR: 0.5, NDCG5: 0.6, MAP: 0.7, LatencyMs: 200},
	}

	summary := AggregateSummary("sgrep", results)

	if summary.Tool != "sgrep" {
		t.Errorf("Tool = %v, want 'sgrep'", summary.Tool)
	}

	if summary.NumQueries != 2 {
		t.Errorf("NumQueries = %v, want 2", summary.NumQueries)
	}

	if !floatEquals(summary.MeanMRR, 0.75) {
		t.Errorf("MeanMRR = %v, want 0.75", summary.MeanMRR)
	}

	if !floatEquals(summary.MeanNDCG5, 0.7) {
		t.Errorf("MeanNDCG5 = %v, want 0.7", summary.MeanNDCG5)
	}

	if !floatEquals(summary.MeanLatencyMs, 150) {
		t.Errorf("MeanLatencyMs = %v, want 150", summary.MeanLatencyMs)
	}
}
