package main

import (
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/search"
)

func TestMeasureUsesAllLabeledJudgmentsForIdealRanking(t *testing.T) {
	query := queryCase{Judgments: []judgment{
		{File: "impl.go", Rel: 3},
		{File: "support.go", Rel: 2},
	}}
	results := []search.Result{
		{FilePath: "README.md"},
		{FilePath: "support.go"},
		{FilePath: "support.go", StartLine: 100},
	}

	got := measure(results, query)
	if got.MRR != 0.5 {
		t.Fatalf("MRR = %v, want 0.5", got.MRR)
	}
	if got.Recall10 != 0.5 {
		t.Fatalf("R@10 = %v, want 0.5", got.Recall10)
	}
	if got.NDCG10 >= 0.5 {
		t.Fatalf("NDCG@10 = %v, want penalty for missing grade-3 implementation", got.NDCG10)
	}
	if got.PromotionErrors != 1 {
		t.Fatalf("promotion errors = %d, want 1", got.PromotionErrors)
	}
}

func TestProbabilityLogitPreservesOrdering(t *testing.T) {
	low := probabilityLogit(0.2)
	high := probabilityLogit(0.8)
	if low >= high {
		t.Fatalf("logit ordering: low=%v high=%v", low, high)
	}
}

func TestScaledProbabilityShrinksTowardNeutral(t *testing.T) {
	if got := scaledProbability(0.8, 0.5); got != 0.65 {
		t.Fatalf("scaled probability = %v, want 0.65", got)
	}
	if got := scaledProbability(0.2, 0); got != 0.5 {
		t.Fatalf("zero-scale probability = %v, want 0.5", got)
	}
}

func TestPercentileNearestRank(t *testing.T) {
	values := []time.Duration{5 * time.Millisecond, time.Millisecond, 3 * time.Millisecond, 2 * time.Millisecond}
	if got := percentile(values, 0.50); got != 2 {
		t.Fatalf("p50 = %vms, want 2ms", got)
	}
	if got := percentile(values, 0.95); got != 5 {
		t.Fatalf("p95 = %vms, want 5ms", got)
	}
}

func TestSameResultsIncludesPathsRangesAndScores(t *testing.T) {
	want := []search.Result{{FilePath: "a.go", StartLine: 10, EndLine: 20, Score: 0.3}}
	if !sameResults(want, append([]search.Result(nil), want...)) {
		t.Fatal("identical results reported different")
	}
	changed := append([]search.Result(nil), want...)
	changed[0].EndLine++
	if sameResults(want, changed) {
		t.Fatal("changed line range reported identical")
	}
}
