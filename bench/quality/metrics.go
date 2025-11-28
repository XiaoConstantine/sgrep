package quality

import (
	"math"
	"sort"
)

// MRR computes Mean Reciprocal Rank.
// Returns 1/rank of the first relevant result, or 0 if none found.
func MRR(rels []Relevance) float64 {
	for i, r := range rels {
		if r > RelNone {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// MRRAtK computes MRR considering only top-k results.
func MRRAtK(rels []Relevance, k int) float64 {
	if k > len(rels) {
		k = len(rels)
	}
	for i := 0; i < k; i++ {
		if rels[i] > RelNone {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// PrecisionAtK computes precision at rank k.
// Precision@k = (relevant items in top-k) / k
func PrecisionAtK(rels []Relevance, k int) float64 {
	if k <= 0 {
		return 0
	}
	if k > len(rels) {
		k = len(rels)
	}
	if k == 0 {
		return 0
	}

	relCount := 0
	for i := 0; i < k; i++ {
		if rels[i] > RelNone {
			relCount++
		}
	}
	return float64(relCount) / float64(k)
}

// RecallAtK computes recall at rank k.
// Recall@k = (relevant items in top-k) / (total relevant items)
func RecallAtK(rels []Relevance, totalRelevant, k int) float64 {
	if totalRelevant == 0 || k <= 0 {
		return 0
	}
	if k > len(rels) {
		k = len(rels)
	}

	relCount := 0
	for i := 0; i < k; i++ {
		if rels[i] > RelNone {
			relCount++
		}
	}
	return float64(relCount) / float64(totalRelevant)
}

// AveragePrecision computes Average Precision (AP).
// AP = sum(P@i * rel_i) / total_relevant, where i is each relevant position.
func AveragePrecision(rels []Relevance, totalRelevant int) float64 {
	if totalRelevant == 0 {
		return 0
	}

	sumPrec := 0.0
	relSeen := 0
	for i, r := range rels {
		if r > RelNone {
			relSeen++
			sumPrec += float64(relSeen) / float64(i+1) // P@i at this position
		}
	}

	if relSeen == 0 {
		return 0
	}
	return sumPrec / float64(totalRelevant)
}

// AveragePrecisionAtK computes AP considering only top-k results.
func AveragePrecisionAtK(rels []Relevance, totalRelevant, k int) float64 {
	if totalRelevant == 0 || k <= 0 {
		return 0
	}
	if k > len(rels) {
		k = len(rels)
	}

	sumPrec := 0.0
	relSeen := 0
	for i := 0; i < k; i++ {
		if rels[i] > RelNone {
			relSeen++
			sumPrec += float64(relSeen) / float64(i+1)
		}
	}

	if relSeen == 0 {
		return 0
	}
	return sumPrec / float64(totalRelevant)
}

// DCG computes Discounted Cumulative Gain at rank k.
// DCG@k = Σ (2^rel_i - 1) / log2(i + 2) for i = 0..k-1
func DCG(rels []Relevance, k int) float64 {
	if k <= 0 {
		return 0
	}
	if k > len(rels) {
		k = len(rels)
	}

	dcg := 0.0
	for i := 0; i < k; i++ {
		rel := float64(rels[i])
		if rel <= 0 {
			continue
		}
		dcg += (math.Pow(2, rel) - 1) / math.Log2(float64(i+2))
	}
	return dcg
}

// NDCG computes Normalized Discounted Cumulative Gain at rank k.
// NDCG@k = DCG@k / IDCG@k, where IDCG is the ideal DCG (sorted by relevance).
func NDCG(rels []Relevance, k int) float64 {
	if k <= 0 || len(rels) == 0 {
		return 0
	}

	dcg := DCG(rels, k)
	if dcg == 0 {
		return 0
	}

	// Build ideal ranking (sort by relevance descending)
	ideal := make([]Relevance, len(rels))
	copy(ideal, rels)
	sort.Slice(ideal, func(i, j int) bool {
		return ideal[i] > ideal[j]
	})

	idcg := DCG(ideal, k)
	if idcg == 0 {
		return 0
	}

	return dcg / idcg
}

// F1 computes F1 score from precision and recall.
func F1(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

// ComputeAllMetrics computes all IR metrics for a single query result.
func ComputeAllMetrics(results []SearchResult, query *QueryCase) *EvalResult {
	relMap := query.RelevanceMap()
	rels := ToRelevances(results, relMap)
	totalRel := query.TotalRelevant()

	return &EvalResult{
		Query:       query.Query,
		MRR:         MRR(rels),
		NDCG5:       NDCG(rels, 5),
		NDCG10:      NDCG(rels, 10),
		MAP:         AveragePrecision(rels, totalRel),
		PrecisionAt5:  PrecisionAtK(rels, 5),
		PrecisionAt10: PrecisionAtK(rels, 10),
		RecallAt5:     RecallAtK(rels, totalRel, 5),
		RecallAt10:    RecallAtK(rels, totalRel, 10),
	}
}

// AggregateSummary computes mean metrics across multiple eval results.
func AggregateSummary(tool string, results []EvalResult) Summary {
	if len(results) == 0 {
		return Summary{Tool: tool}
	}

	var sum Summary
	sum.Tool = tool
	sum.NumQueries = len(results)

	for _, r := range results {
		sum.MeanMRR += r.MRR
		sum.MeanNDCG5 += r.NDCG5
		sum.MeanNDCG10 += r.NDCG10
		sum.MeanMAP += r.MAP
		sum.MeanP5 += r.PrecisionAt5
		sum.MeanP10 += r.PrecisionAt10
		sum.MeanR5 += r.RecallAt5
		sum.MeanR10 += r.RecallAt10
		sum.MeanLatencyMs += r.LatencyMs
		sum.TotalTokens += r.TokensUsed
	}

	n := float64(len(results))
	sum.MeanMRR /= n
	sum.MeanNDCG5 /= n
	sum.MeanNDCG10 /= n
	sum.MeanMAP /= n
	sum.MeanP5 /= n
	sum.MeanP10 /= n
	sum.MeanR5 /= n
	sum.MeanR10 /= n
	sum.MeanLatencyMs /= n

	return sum
}

// JaccardSimilarity computes Jaccard similarity between two file sets.
// Useful for measuring stability across query paraphrases.
func JaccardSimilarity(set1, set2 []string) float64 {
	if len(set1) == 0 && len(set2) == 0 {
		return 1.0
	}

	m1 := make(map[string]bool, len(set1))
	for _, s := range set1 {
		m1[s] = true
	}

	intersection := 0
	union := len(set1)

	for _, s := range set2 {
		if m1[s] {
			intersection++
		} else {
			union++
		}
	}

	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// Diversity computes the ratio of unique files to total results.
// Higher diversity means fewer duplicate files in results.
func Diversity(results []SearchResult) float64 {
	if len(results) == 0 {
		return 0
	}

	seen := make(map[string]bool)
	for _, r := range results {
		seen[r.File] = true
	}

	return float64(len(seen)) / float64(len(results))
}
