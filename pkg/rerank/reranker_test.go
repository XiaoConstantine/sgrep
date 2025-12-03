package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestReranker_Rerank(t *testing.T) {
	// Create a mock server that simulates llama.cpp /v1/rerank endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		if r.Method != "POST" {
			t.Errorf("unexpected method: %s", r.Method)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse request
		var req rerankRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Validate request
		if req.Query == "" {
			http.Error(w, "query required", http.StatusBadRequest)
			return
		}

		// Create mock response - simulate ranking based on document order
		results := make([]RerankResult, len(req.Documents))
		for i := range req.Documents {
			// Give higher scores to earlier documents (simulate relevance ranking)
			results[i] = RerankResult{
				Index: i,
				Score: 1.0 - float64(i)*0.1,
			}
		}

		resp := rerankResponse{
			Model:   "mock-reranker",
			Results: results,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create reranker with mock server
	reranker := NewWithConfig(Config{
		Endpoint:  server.URL,
		Timeout:   5 * time.Second,
		AutoStart: false, // Don't try to start real server
	})

	// Test reranking
	ctx := context.Background()
	query := "test query"
	documents := []string{"doc1", "doc2", "doc3"}

	results, err := reranker.Rerank(ctx, query, documents)
	if err != nil {
		t.Fatalf("Rerank failed: %v", err)
	}

	if len(results) != len(documents) {
		t.Errorf("expected %d results, got %d", len(documents), len(results))
	}

	// Verify first document has highest score
	if len(results) > 0 && results[0].Score < 0.9 {
		t.Errorf("expected first document to have high score, got %f", results[0].Score)
	}
}

func TestReranker_RerankTopK(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rerankRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Return results with varying scores
		results := []RerankResult{
			{Index: 0, Score: 0.5},
			{Index: 1, Score: 0.9}, // Highest
			{Index: 2, Score: 0.3},
			{Index: 3, Score: 0.7},
			{Index: 4, Score: 0.1},
		}

		_ = json.NewEncoder(w).Encode(rerankResponse{Results: results})
	}))
	defer server.Close()

	reranker := NewWithConfig(Config{
		Endpoint:  server.URL,
		Timeout:   5 * time.Second,
		AutoStart: false,
	})

	ctx := context.Background()
	documents := []string{"a", "b", "c", "d", "e"}

	// Get top 2
	results, err := reranker.RerankTopK(ctx, "query", documents, 2)
	if err != nil {
		t.Fatalf("RerankTopK failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// Verify results are sorted by score descending
	if len(results) >= 2 {
		if results[0].Score < results[1].Score {
			t.Errorf("results not sorted by score: %f < %f", results[0].Score, results[1].Score)
		}
		// Index 1 should be first (highest score 0.9)
		if results[0].Index != 1 {
			t.Errorf("expected index 1 to be first, got index %d", results[0].Index)
		}
	}
}

func TestReranker_EmptyDocuments(t *testing.T) {
	reranker := NewWithConfig(Config{
		Endpoint:  "http://localhost:9999", // Won't be called
		Timeout:   5 * time.Second,
		AutoStart: false,
	})

	ctx := context.Background()
	results, err := reranker.Rerank(ctx, "query", []string{})

	if err != nil {
		t.Errorf("expected no error for empty documents, got: %v", err)
	}

	if results != nil {
		t.Errorf("expected nil results for empty documents, got: %v", results)
	}
}

func TestReranker_ServerError(t *testing.T) {
	// Create server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	reranker := NewWithConfig(Config{
		Endpoint:  server.URL,
		Timeout:   5 * time.Second,
		AutoStart: false,
	})

	ctx := context.Background()
	_, err := reranker.Rerank(ctx, "query", []string{"doc1"})

	if err == nil {
		t.Error("expected error for server error response")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Endpoint != "http://localhost:8081" {
		t.Errorf("expected default endpoint http://localhost:8081, got %s", cfg.Endpoint)
	}

	if cfg.Timeout != 30*time.Second {
		t.Errorf("expected default timeout 30s, got %v", cfg.Timeout)
	}

	if !cfg.AutoStart {
		t.Error("expected AutoStart to be true by default")
	}
}

func TestRerankerAvailable(t *testing.T) {
	// This will return false in test environment (no model downloaded)
	available := RerankerAvailable()
	// Just verify it doesn't panic
	_ = available
}
