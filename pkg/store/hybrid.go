package store

import "sort"

// fuseHybridCandidates combines independently ranked dense and lexical lists
// with weighted reciprocal-rank fusion. RRF avoids treating backend-specific
// cosine distances and BM25 magnitudes as directly comparable.
func fuseHybridCandidates(dense []DenseSearchResult, lexical []BM25SearchResult, limit int, semanticWeight, bm25Weight float64) []DenseSearchResult {
	if limit <= 0 {
		return nil
	}
	totalWeight := semanticWeight + bm25Weight
	if totalWeight <= 0 {
		semanticWeight, bm25Weight, totalWeight = 0.6, 0.4, 1
	}
	semanticWeight /= totalWeight
	bm25Weight /= totalWeight

	type candidate struct {
		id    string
		score float64
	}
	candidates := make(map[string]*candidate, len(dense)+len(lexical))
	get := func(id string) *candidate {
		c := candidates[id]
		if c == nil {
			c = &candidate{id: id}
			candidates[id] = c
		}
		return c
	}

	const rrfK = 60.0
	seenDense := make(map[string]struct{}, len(dense))
	for rank, hit := range dense {
		if hit.ID == "" {
			continue
		}
		if _, ok := seenDense[hit.ID]; ok {
			continue
		}
		seenDense[hit.ID] = struct{}{}
		get(hit.ID).score += semanticWeight / (rrfK + float64(rank+1))
	}
	seenLexical := make(map[string]struct{}, len(lexical))
	for rank, hit := range lexical {
		if hit.ID == "" {
			continue
		}
		if _, ok := seenLexical[hit.ID]; ok {
			continue
		}
		seenLexical[hit.ID] = struct{}{}
		get(hit.ID).score += bm25Weight / (rrfK + float64(rank+1))
	}

	ranked := make([]candidate, 0, len(candidates))
	for _, c := range candidates {
		ranked = append(ranked, *c)
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].id < ranked[j].id
	})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	results := make([]DenseSearchResult, len(ranked))
	for i, item := range ranked {
		// Preserve the Storer convention that lower scores rank first.
		results[i] = DenseSearchResult{ID: item.id, Distance: -item.score}
	}
	return results
}
