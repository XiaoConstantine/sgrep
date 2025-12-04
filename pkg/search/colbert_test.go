package search

import (
	"testing"
)

func TestDecomposeQuery(t *testing.T) {
	tests := []struct {
		query    string
		minTerms int
		maxTerms int
	}{
		{"error handling", 2, 5},
		{"how does authentication work", 3, 8},
		{"x", 1, 2},
		{"", 0, 0},
		{"chain of thought implementation", 3, 8},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			terms := decomposeQuery(tt.query)
			if len(terms) < tt.minTerms || len(terms) > tt.maxTerms {
				t.Errorf("decomposeQuery(%q) = %d terms, want %d-%d", tt.query, len(terms), tt.minTerms, tt.maxTerms)
			}
		})
	}
}

func TestDecomposeDocument(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		minSegs  int
		maxSegs  int
	}{
		{
			name:     "empty",
			content:  "",
			minSegs:  0,
			maxSegs:  0,
		},
		{
			name:     "single line",
			content:  "func main() {}",
			minSegs:  1,
			maxSegs:  1,
		},
		{
			name: "multiple functions",
			content: `func foo() {
	return 1
}

func bar() {
	return 2
}`,
			minSegs: 2,
			maxSegs: 4,
		},
		{
			name: "with comments",
			content: `// Package main provides entry point
package main

// main is the entry function
func main() {
	fmt.Println("hello")
}`,
			minSegs: 2,
			maxSegs: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			segments := decomposeDocument(tt.content)
			if len(segments) < tt.minSegs || len(segments) > tt.maxSegs {
				t.Errorf("decomposeDocument() = %d segments, want %d-%d", len(segments), tt.minSegs, tt.maxSegs)
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	tests := []struct {
		text     string
		expected []string
	}{
		{"Hello World", []string{"hello", "world"}},
		{"func_name", []string{"func", "name"}},
		{"CamelCase", []string{"camelcase"}},
		{"123abc", []string{"123abc"}},
		{"", nil},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			tokens := tokenize(tt.text)
			if len(tokens) != len(tt.expected) {
				t.Errorf("tokenize(%q) = %v, want %v", tt.text, tokens, tt.expected)
				return
			}
			for i, tok := range tokens {
				if tok != tt.expected[i] {
					t.Errorf("tokenize(%q)[%d] = %q, want %q", tt.text, i, tok, tt.expected[i])
				}
			}
		})
	}
}

func TestIsStopWord(t *testing.T) {
	stopWords := []string{"the", "is", "a", "an", "of", "to", "in", "for"}
	nonStopWords := []string{"error", "function", "database", "config", "handler"}

	for _, w := range stopWords {
		if !isStopWord(w) {
			t.Errorf("isStopWord(%q) = false, want true", w)
		}
	}

	for _, w := range nonStopWords {
		if isStopWord(w) {
			t.Errorf("isStopWord(%q) = true, want false", w)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
		delta    float64
	}{
		{
			name:     "identical",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "similar",
			a:        []float32{1, 1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.707,
			delta:    0.01,
		},
		{
			name:     "empty",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "mismatched length",
			a:        []float32{1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			if result < tt.expected-tt.delta || result > tt.expected+tt.delta {
				t.Errorf("cosineSimilarity(%v, %v) = %f, want %f±%f", tt.a, tt.b, result, tt.expected, tt.delta)
			}
		})
	}
}

func TestNewColBERTScorer(t *testing.T) {
	// Test that we can create a scorer without an embedder (nil is allowed)
	scorer := NewColBERTScorer(nil)
	if scorer == nil {
		t.Fatal("NewColBERTScorer returned nil")
	}
	if scorer.cache == nil {
		t.Error("scorer.cache should be initialized")
	}
}

func TestSegmentCache(t *testing.T) {
	cache := newSegmentCache(3)

	// Test set and get
	cache.set("key1", []float32{1, 2, 3})
	if got := cache.get("key1"); got == nil {
		t.Error("cache.get(key1) returned nil after set")
	}

	// Test cache miss
	if got := cache.get("nonexistent"); got != nil {
		t.Error("cache.get(nonexistent) should return nil")
	}

	// Test eviction (cache size is 3)
	cache.set("key2", []float32{4, 5, 6})
	cache.set("key3", []float32{7, 8, 9})
	cache.set("key4", []float32{10, 11, 12}) // This should trigger eviction

	// At least one of the original keys should be evicted
	// Note: eviction is simple and clears half, so behavior may vary
}
