//go:build goexperiment.simd && !arm64

package util

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestSIMDDotProductPreservesTopRanking(t *testing.T) {
	const (
		dims = 768
		docs = 512
		topK = 50
		seed = 47
	)
	rng := rand.New(rand.NewSource(seed))
	query := randomNormalizedVector(rng, dims)

	type scoredIndex struct {
		index int
		score float64
	}
	simdRanking := make([]scoredIndex, docs)
	scalarRanking := make([]scoredIndex, docs)
	for i := range docs {
		doc := randomNormalizedVector(rng, dims)
		simdScore := dotProductFloat32(query, doc)
		scalarScore := dotProductUnrolled8Scalar(query, doc)
		if diff := math.Abs(simdScore - scalarScore); diff > 1e-6 {
			t.Fatalf("document %d score differs by %g: SIMD %.12f scalar %.12f", i, diff, simdScore, scalarScore)
		}
		simdRanking[i] = scoredIndex{index: i, score: simdScore}
		scalarRanking[i] = scoredIndex{index: i, score: scalarScore}
	}

	less := func(ranking []scoredIndex) func(int, int) bool {
		return func(i, j int) bool {
			if ranking[i].score == ranking[j].score {
				return ranking[i].index < ranking[j].index
			}
			return ranking[i].score > ranking[j].score
		}
	}
	sort.Slice(simdRanking, less(simdRanking))
	sort.Slice(scalarRanking, less(scalarRanking))
	for rank := range topK {
		if simdRanking[rank].index != scalarRanking[rank].index {
			t.Fatalf("rank %d differs: SIMD document %d, scalar document %d", rank, simdRanking[rank].index, scalarRanking[rank].index)
		}
	}
}

func randomNormalizedVector(rng *rand.Rand, dims int) []float32 {
	values := make([]float32, dims)
	for i := range values {
		values[i] = rng.Float32()*2 - 1
	}
	return NormalizeVector(values)
}
