package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

// Config holds reranker configuration.
type Config struct {
	Endpoint  string        // Server endpoint (default: http://localhost:8081)
	Timeout   time.Duration // Request timeout (default: 30s)
	AutoStart bool          // Auto-start server if not running (default: true)
	ServerMgr *RerankerManager
}

// DefaultConfig returns sensible defaults for reranker configuration.
func DefaultConfig() Config {
	return Config{
		Endpoint:  "http://localhost:8081",
		Timeout:   30 * time.Second,
		AutoStart: true,
	}
}

// Reranker handles document reranking via llama.cpp server.
type Reranker struct {
	endpoint   string
	client     *http.Client
	serverMgr  *RerankerManager
	autoStart  bool
	startOnce  sync.Once
	startError error
}

// RerankResult contains the reranked document index and score.
type RerankResult struct {
	Index int     `json:"index"`
	Score float64 `json:"relevance_score"`
}

// rerankRequest matches llama.cpp /v1/rerank endpoint format.
type rerankRequest struct {
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// rerankResponse from llama.cpp server.
type rerankResponse struct {
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
}

// New creates a new reranker with default configuration.
func New() (*Reranker, error) {
	mgr, err := NewRerankerManager()
	if err != nil {
		return nil, err
	}

	cfg := DefaultConfig()
	cfg.ServerMgr = mgr
	cfg.Endpoint = mgr.Endpoint()

	return NewWithConfig(cfg), nil
}

// NewWithConfig creates a reranker with custom configuration.
func NewWithConfig(cfg Config) *Reranker {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = "http://localhost:8081"
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &Reranker{
		endpoint:  endpoint,
		client:    &http.Client{Timeout: timeout},
		serverMgr: cfg.ServerMgr,
		autoStart: cfg.AutoStart,
	}
}

// ensureServer starts the reranker server if auto-start is enabled.
func (r *Reranker) ensureServer() error {
	if !r.autoStart || r.serverMgr == nil {
		return nil
	}

	r.startOnce.Do(func() {
		if !r.serverMgr.IsRunning() {
			r.startError = r.serverMgr.Start()
		}
	})

	return r.startError
}

// Rerank takes a query and list of document contents, returns reranked results.
// Documents are reranked by relevance to the query. Higher scores indicate more relevance.
func (r *Reranker) Rerank(ctx context.Context, query string, documents []string) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Ensure server is running
	if err := r.ensureServer(); err != nil {
		return nil, fmt.Errorf("failed to start reranker server: %w", err)
	}

	// Build request
	reqBody, err := json.Marshal(rerankRequest{
		Query:     query,
		Documents: documents,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rerank request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST",
		r.endpoint+"/v1/rerank", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send request
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("rerank returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse rerank response: %w", err)
	}

	return result.Results, nil
}

// RerankTopK reranks documents and returns only the top K results.
func (r *Reranker) RerankTopK(ctx context.Context, query string, documents []string, k int) ([]RerankResult, error) {
	results, err := r.Rerank(ctx, query, documents)
	if err != nil {
		return nil, err
	}

	// Sort by score descending (higher = more relevant)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit to top K
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// IsAvailable checks if reranking is available (model downloaded and server can start).
func (r *Reranker) IsAvailable() bool {
	if r.serverMgr == nil {
		return false
	}

	// Check if model exists
	if !r.serverMgr.ModelExists() {
		return false
	}

	// Check if server is running or can start
	if r.serverMgr.IsRunning() {
		return true
	}

	// Try to start server
	if err := r.serverMgr.Start(); err != nil {
		return false
	}

	return true
}

// Stop stops the reranker server if it was started by this reranker.
func (r *Reranker) Stop() error {
	if r.serverMgr == nil {
		return nil
	}
	return r.serverMgr.Stop()
}

// RerankerAvailable checks if reranking is available without creating a full Reranker.
func RerankerAvailable() bool {
	mgr, err := NewRerankerManager()
	if err != nil {
		return false
	}
	return mgr.ModelExists()
}

// PrintStatus prints reranker status to stderr for debugging.
func (r *Reranker) PrintStatus() {
	if r.serverMgr == nil {
		fmt.Fprintln(os.Stderr, "Reranker: no server manager")
		return
	}

	running, pid, port := r.serverMgr.Status()
	if running {
		fmt.Fprintf(os.Stderr, "Reranker: running on port %d", port)
		if pid > 0 {
			fmt.Fprintf(os.Stderr, " (PID %d)", pid)
		}
		fmt.Fprintln(os.Stderr)
	} else {
		fmt.Fprintln(os.Stderr, "Reranker: not running")
	}

	if r.serverMgr.ModelExists() {
		fmt.Fprintf(os.Stderr, "Model: %s\n", r.serverMgr.ModelPath())
	} else {
		fmt.Fprintln(os.Stderr, "Model: not downloaded (run 'sgrep setup --with-rerank')")
	}
}
