package main

import (
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/search"
)

type judgment struct {
	File string `json:"file"`
	Rel  int    `json:"rel"`
}

type queryCase struct {
	Split        string     `json:"split"`
	Query        string     `json:"query"`
	AnswersQuery bool       `json:"answers_query"`
	Judgments    []judgment `json:"judgments"`
}

type dataset struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Corpus      string      `json:"corpus"`
	CorpusHash  string      `json:"corpus_hash"`
	Cases       []queryCase `json:"cases"`
}

type queryMetrics struct {
	MRR             float64 `json:"mrr"`
	NDCG10          float64 `json:"ndcg@10"`
	Recall10        float64 `json:"recall@10"`
	PromotionErrors int     `json:"implementation_vs_prose_promotion_errors"`
}

func measure(results []search.Result, query queryCase) queryMetrics {
	rels := make(map[string]int, len(query.Judgments))
	ideal := make([]int, 0, len(query.Judgments))
	for _, item := range query.Judgments {
		rels[filepath.ToSlash(item.File)] = item.Rel
		ideal = append(ideal, item.Rel)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(ideal)))

	metrics := queryMetrics{}
	var ranked []int
	relevantSeen := 0
	firstImplementation := -1
	proseBeforeImplementation := 0
	seenFiles := make(map[string]struct{}, len(results))
	rank := 0
	for _, result := range results {
		path := filepath.ToSlash(result.FilePath)
		if _, seen := seenFiles[path]; seen {
			continue
		}
		seenFiles[path] = struct{}{}
		if rank >= 10 {
			break
		}
		rel := rels[path]
		ranked = append(ranked, rel)
		if rel > 0 {
			relevantSeen++
			if metrics.MRR == 0 {
				metrics.MRR = 1 / float64(rank+1)
			}
			if firstImplementation == -1 {
				firstImplementation = rank
				metrics.PromotionErrors = proseBeforeImplementation
			}
		} else if isProseOrTest(path) && firstImplementation == -1 {
			proseBeforeImplementation++
		}
		rank++
	}
	if len(query.Judgments) > 0 {
		metrics.Recall10 = float64(relevantSeen) / float64(len(query.Judgments))
		metrics.NDCG10 = dcg(ranked, 10) / dcg(ideal, 10)
	}
	return metrics
}

func dcg(rels []int, k int) float64 {
	total := 0.0
	for i := 0; i < min(k, len(rels)); i++ {
		total += (math.Pow(2, float64(rels[i])) - 1) / math.Log2(float64(i+2))
	}
	return total
}

func isProseOrTest(path string) bool {
	lower := strings.ToLower(filepath.ToSlash(path))
	return strings.HasSuffix(lower, ".md") ||
		strings.Contains(lower, "/docs/") || strings.HasPrefix(lower, "docs/") ||
		strings.Contains(lower, "/examples/") || strings.HasPrefix(lower, "examples/") ||
		strings.HasSuffix(lower, "_test.go") || strings.Contains(lower, "compatibility_test/")
}

func percentile(durations []time.Duration, percentile float64) float64 {
	if len(durations) == 0 {
		return 0
	}
	values := append([]time.Duration(nil), durations...)
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	index := int(math.Ceil(percentile*float64(len(values)))) - 1
	index = max(0, min(len(values)-1, index))
	return float64(values[index].Microseconds()) / 1000
}
