package store

import "sort"

func fuseHybridCandidates(dense []DenseSearchResult, lexical []BM25SearchResult, limit int, semanticWeight, bm25Weight float64) []DenseSearchResult {
	if limit <= 0 {
		return nil
	}
	totalWeight := semanticWeight + bm25Weight
	if totalWeight <= 0 {
		semanticWeight = 0.6
		bm25Weight = 0.4
		totalWeight = 1
	}
	semanticWeight /= totalWeight
	bm25Weight /= totalWeight

	type candidate struct {
		id            string
		denseDistance float64
		hasDense      bool
		bm25Score     float64
		hasBM25       bool
	}
	candidates := make(map[string]*candidate, len(dense)+len(lexical))

	for _, hit := range dense {
		if hit.ID == "" {
			continue
		}
		c := candidates[hit.ID]
		if c == nil {
			c = &candidate{id: hit.ID, denseDistance: hit.Distance, hasDense: true}
			candidates[hit.ID] = c
		} else if !c.hasDense || hit.Distance < c.denseDistance {
			c.denseDistance = hit.Distance
			c.hasDense = true
		}
	}
	bestBM25 := 0.0
	worstBM25 := 0.0
	haveBM25 := false
	for _, hit := range lexical {
		if hit.ID == "" {
			continue
		}
		c := candidates[hit.ID]
		if c == nil {
			c = &candidate{id: hit.ID}
			candidates[hit.ID] = c
		}
		if !c.hasBM25 || hit.Score < c.bm25Score {
			c.bm25Score = hit.Score
			c.hasBM25 = true
		}
		if !haveBM25 {
			bestBM25 = hit.Score
			worstBM25 = hit.Score
			haveBM25 = true
		} else {
			if hit.Score < bestBM25 {
				bestBM25 = hit.Score
			}
			if hit.Score > worstBM25 {
				worstBM25 = hit.Score
			}
		}
	}

	results := make([]DenseSearchResult, 0, len(candidates))
	for _, c := range candidates {
		denseScore := 0.0
		if c.hasDense {
			denseScore = 1 - c.denseDistance
			if denseScore < 0 {
				denseScore = 0
			} else if denseScore > 1 {
				denseScore = 1
			}
		}
		lexicalScore := 0.0
		if c.hasBM25 {
			if bestBM25 == worstBM25 {
				lexicalScore = 1
			} else {
				lexicalScore = (worstBM25 - c.bm25Score) / (worstBM25 - bestBM25)
			}
		}
		score := semanticWeight*denseScore + bm25Weight*lexicalScore
		results = append(results, DenseSearchResult{
			ID:       c.id,
			Distance: 1 - score,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Distance != results[j].Distance {
			return results[i].Distance < results[j].Distance
		}
		return results[i].ID < results[j].ID
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}
