package search

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/util"
)

const benchDims = 768

// Helper to generate random normalized vector
func randomNormalizedVec(dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1
	}
	return util.NormalizeVector(vec)
}

// Helper to generate random unnormalized vector
func randomVec(dims int) []float32 {
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1
	}
	return vec
}

// ============================================================
// Similarity Function Benchmarks
// ============================================================

func BenchmarkSimilarity(b *testing.B) {
	// Setup: create random vectors
	unnormA := randomVec(benchDims)
	unnormB := randomVec(benchDims)
	normA := randomNormalizedVec(benchDims)
	normB := randomNormalizedVec(benchDims)

	b.Run("CosineSimilarity_768dims", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cosineSimilarity(unnormA, unnormB)
		}
	})

	b.Run("DotProductSimilarity_768dims", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dotProductSimilarity(normA, normB)
		}
	})

	b.Run("DotProductUnrolled8_768dims", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			util.DotProductUnrolled8(normA, normB)
		}
	})
}

// ============================================================
// MaxSim Computation Benchmarks
// ============================================================

func BenchmarkMaxSim(b *testing.B) {
	setupDocEmbs := func(numDocs int) [][]float32 {
		embs := make([][]float32, numDocs)
		for i := range embs {
			embs[i] = randomNormalizedVec(benchDims)
		}
		return embs
	}

	for _, numSegs := range []int{5, 10, 15} {
		b.Run(fmt.Sprintf("Sequential_%dsegs", numSegs), func(b *testing.B) {
			qEmb := randomNormalizedVec(benchDims)
			docEmbs := setupDocEmbs(numSegs)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				maxSim := float64(-1)
				for _, dEmb := range docEmbs {
					sim := dotProductSimilarity(qEmb, dEmb)
					if sim > maxSim {
						maxSim = sim
					}
				}
				_ = maxSim
			}
		})

		b.Run(fmt.Sprintf("Batch_%dsegs", numSegs), func(b *testing.B) {
			qEmb := randomNormalizedVec(benchDims)
			docEmbs := setupDocEmbs(numSegs)
			distances := make([]float64, numSegs)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = maxSimBatch(qEmb, docEmbs, distances)
			}
		})
	}
}

// ============================================================
// Full Scoring Benchmarks (varying document and query term counts)
// ============================================================

func BenchmarkScoring(b *testing.B) {
	// Setup: pre-compute embeddings to isolate scoring logic
	setupQueryDocEmbs := func(numQueryTerms, numDocsPerDoc, numDocs int) ([][]float32, [][][]float32) {
		queryEmbs := make([][]float32, numQueryTerms)
		for i := range queryEmbs {
			queryEmbs[i] = randomNormalizedVec(benchDims)
		}

		// Each "document" has numDocsPerDoc segments
		docEmbs := make([][][]float32, numDocs)
		for d := range docEmbs {
			docEmbs[d] = make([][]float32, numDocsPerDoc)
			for s := range docEmbs[d] {
				docEmbs[d][s] = randomNormalizedVec(benchDims)
			}
		}
		return queryEmbs, docEmbs
	}

	// Sequential scoring function (original approach)
	scoreDocsSequential := func(queryEmbs [][]float32, docEmbs [][][]float32) []float64 {
		scores := make([]float64, len(docEmbs))

		for d, segs := range docEmbs {
			var totalScore float64
			for _, qEmb := range queryEmbs {
				maxSim := float64(-1)
				for _, dEmb := range segs {
					sim := dotProductSimilarity(qEmb, dEmb)
					if sim > maxSim {
						maxSim = sim
					}
				}
				if maxSim > 0 {
					totalScore += maxSim
				}
			}
			scores[d] = totalScore / float64(len(queryEmbs))
		}
		return scores
	}

	// Optimized scoring with maxSimBatch
	scoreDocsOptimized := func(queryEmbs [][]float32, docEmbs [][][]float32) []float64 {
		scores := make([]float64, len(docEmbs))
		maxSegments := 0
		for _, segs := range docEmbs {
			if len(segs) > maxSegments {
				maxSegments = len(segs)
			}
		}
		distances := make([]float64, maxSegments)

		for d, segs := range docEmbs {
			var totalScore float64
			for _, qEmb := range queryEmbs {
				maxSim := maxSimBatch(qEmb, segs, distances[:len(segs)])
				if maxSim > 0 {
					totalScore += maxSim
				}
			}
			scores[d] = totalScore / float64(len(queryEmbs))
		}
		return scores
	}

	// Test matrix: docs x query terms
	for _, numDocs := range []int{10, 25, 50} {
		for _, numTerms := range []int{2, 4, 8} {
			name := fmt.Sprintf("Docs%d_Terms%d", numDocs, numTerms)
			queryEmbs, docEmbs := setupQueryDocEmbs(numTerms, 10, numDocs)

			b.Run(name+"_Sequential", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = scoreDocsSequential(queryEmbs, docEmbs)
				}
			})

			b.Run(name+"_Optimized", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = scoreDocsOptimized(queryEmbs, docEmbs)
				}
			})
		}
	}
}

// ============================================================
// Normalization Overhead Benchmark
// ============================================================

func BenchmarkNormalization(b *testing.B) {
	vec := randomVec(benchDims)

	b.Run("NormalizeVector_InPlace", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Make a copy since NormalizeVector is in-place
			v := make([]float32, benchDims)
			copy(v, vec)
			util.NormalizeVector(v)
		}
	})

	b.Run("NormalizeVectorCopy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = util.NormalizeVectorCopy(vec)
		}
	})
}

// ============================================================
// Cache Benchmarks
// ============================================================

func BenchmarkSegmentCache(b *testing.B) {
	emb := randomVec(benchDims)

	b.Run("Set_WithNormalization", func(b *testing.B) {
		cache := newSegmentCache(10000)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.set(fmt.Sprintf("key%d", i%10000), emb)
		}
	})

	b.Run("Get_Hit", func(b *testing.B) {
		cache := newSegmentCache(1000)
		cache.set("testkey", emb)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = cache.get("testkey")
		}
	})

	b.Run("Get_Miss", func(b *testing.B) {
		cache := newSegmentCache(1000)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = cache.get("nonexistent")
		}
	})
}

// ============================================================
// Query/Document Decomposition Benchmarks
// ============================================================

func BenchmarkDecomposition(b *testing.B) {
	shortQuery := "error handling"
	longQuery := "how does the authentication middleware handle JWT token validation"
	shortDoc := `func main() {
	fmt.Println("hello")
}`
	longDoc := `// Package auth provides authentication middleware
package auth

import (
	"net/http"
	"github.com/golang-jwt/jwt/v5"
)

// Middleware validates JWT tokens
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		// Validate token
		claims, err := validateToken(token)
		if err != nil {
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}
		// Store claims in context
		ctx := context.WithValue(r.Context(), "claims", claims)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func validateToken(token string) (*jwt.Claims, error) {
	// Token validation logic
	return nil, nil
}`

	b.Run("DecomposeQuery_Short", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = decomposeQuery(shortQuery)
		}
	})

	b.Run("DecomposeQuery_Long", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = decomposeQuery(longQuery)
		}
	})

	b.Run("DecomposeDocument_Short", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = decomposeDocument(shortDoc)
		}
	})

	b.Run("DecomposeDocument_Long", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = decomposeDocument(longDoc)
		}
	})
}
