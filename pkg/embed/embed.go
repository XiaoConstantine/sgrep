package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/server"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

const (
	defaultEndpoint  = "http://localhost:8080"
	defaultTimeout   = 30 * time.Second
	maxContextTokens = 1500 // Safe limit for 2048 context (llama.cpp tokens)
)

// Config holds embedder configuration (dependency injection).
type Config struct {
	Endpoint   string
	Timeout    time.Duration
	CacheSize  int
	AutoStart  bool
	ServerMgr  *server.Manager
	EventBox   *util.EventBox
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	endpoint := os.Getenv("SGREP_ENDPOINT")
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	return Config{
		Endpoint:  endpoint,
		Timeout:   defaultTimeout,
		CacheSize: 10000,
		AutoStart: true,
	}
}

// Embedder generates embeddings via llama.cpp server.
type Embedder struct {
	endpoint   string
	client     *http.Client
	cache      *Cache
	mu         sync.Mutex
	serverMgr  *server.Manager
	autoStart  bool
	startOnce  sync.Once
	startError error
	eventBox   *util.EventBox

	// Stats
	totalRequests int64
	cacheHits     int64
	errors        int64
}

// New creates a new embedder with auto-start enabled.
func New() *Embedder {
	return NewWithConfig(DefaultConfig())
}

// NewWithOptions creates a new embedder with configurable auto-start.
// Deprecated: Use NewWithConfig for full control.
func NewWithOptions(autoStart bool) *Embedder {
	cfg := DefaultConfig()
	cfg.AutoStart = autoStart
	return NewWithConfig(cfg)
}

// NewWithConfig creates an embedder with full configuration control.
// This is the preferred constructor for dependency injection.
func NewWithConfig(cfg Config) *Embedder {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = defaultEndpoint
	}

	var mgr *server.Manager
	if cfg.AutoStart {
		if cfg.ServerMgr != nil {
			mgr = cfg.ServerMgr
		} else {
			mgr, _ = server.NewManager()
		}
		if mgr != nil {
			endpoint = mgr.Endpoint()
		}
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	cacheSize := cfg.CacheSize
	if cacheSize == 0 {
		cacheSize = 10000
	}

	return &Embedder{
		endpoint:  endpoint,
		client:    &http.Client{Timeout: timeout},
		cache:     NewCache(cacheSize),
		serverMgr: mgr,
		autoStart: cfg.AutoStart,
		eventBox:  cfg.EventBox,
	}
}

// Embed generates an embedding for the given text.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	// Ensure server is running (once per embedder lifetime)
	if err := e.ensureServer(); err != nil {
		return nil, err
	}

	e.mu.Lock()
	e.totalRequests++
	e.mu.Unlock()

	// Truncate if too long to avoid context size errors
	text = truncateToTokenLimit(text, maxContextTokens)

	// Check cache
	if cached := e.cache.Get(text); cached != nil {
		e.mu.Lock()
		e.cacheHits++
		e.mu.Unlock()
		return cached, nil
	}

	// Call llama.cpp /embedding endpoint
	embedding, err := e.callLlamaCpp(ctx, text)
	if err != nil {
		e.mu.Lock()
		e.errors++
		e.mu.Unlock()
		return nil, err
	}

	// Cache result
	e.cache.Set(text, embedding)

	return embedding, nil
}

// ensureServer starts the embedding server if auto-start is enabled.
func (e *Embedder) ensureServer() error {
	if !e.autoStart || e.serverMgr == nil {
		return nil
	}

	e.startOnce.Do(func() {
		if !e.serverMgr.IsRunning() {
			// Emit event if eventBox is configured
			if e.eventBox != nil {
				e.eventBox.Set(util.EvtServerStarting, nil)
			}

			fmt.Fprintln(os.Stderr, "Starting embedding server...")
			e.startError = e.serverMgr.Start()

			if e.startError == nil {
				fmt.Fprintln(os.Stderr, "Embedding server started")
				if e.eventBox != nil {
					e.eventBox.Set(util.EvtServerReady, nil)
				}
			} else {
				if e.eventBox != nil {
					e.eventBox.Set(util.EvtServerError, e.startError)
				}
			}
		}
	})

	return e.startError
}

// truncateToTokenLimit truncates text to stay under the token limit.
// Uses ~3 chars per token (conservative for code which has more special chars).
func truncateToTokenLimit(text string, maxTokens int) string {
	maxChars := maxTokens * 3
	if len(text) <= maxChars {
		return text
	}
	// Truncate at word/line boundary
	truncated := text[:maxChars]
	if idx := strings.LastIndex(truncated, "\n"); idx > maxChars*3/4 {
		truncated = truncated[:idx]
	} else if idx := strings.LastIndex(truncated, " "); idx > maxChars/2 {
		truncated = truncated[:idx]
	}
	return truncated
}

// EmbedBatch generates embeddings for multiple texts using true batch API.
// This sends all texts in a single HTTP request for maximum efficiency.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Ensure server is running
	if err := e.ensureServer(); err != nil {
		return nil, err
	}

	// Truncate all texts first
	truncatedTexts := make([]string, len(texts))
	for i, text := range texts {
		truncatedTexts[i] = truncateToTokenLimit(text, maxContextTokens)
	}

	// Check cache for all texts first
	results := make([][]float32, len(texts))
	uncachedIndices := make([]int, 0, len(texts))
	uncachedTexts := make([]string, 0, len(texts))

	for i, text := range truncatedTexts {
		if cached := e.cache.Get(text); cached != nil {
			results[i] = cached
			e.mu.Lock()
			e.cacheHits++
			e.mu.Unlock()
		} else {
			uncachedIndices = append(uncachedIndices, i)
			uncachedTexts = append(uncachedTexts, text)
		}
	}

	// If all cached, return early
	if len(uncachedTexts) == 0 {
		return results, nil
	}

	// Try true batch API first (single request for all texts)
	embeddings, err := e.callLlamaCppBatch(ctx, uncachedTexts)
	if err != nil {
		// Fall back to individual requests if batch fails
		return e.embedBatchFallback(ctx, texts)
	}

	// Store results and cache them
	for i, idx := range uncachedIndices {
		results[idx] = embeddings[i]
		e.cache.Set(uncachedTexts[i], embeddings[i])
	}

	return results, nil
}

// callLlamaCppBatch sends multiple texts in a single request
func (e *Embedder) callLlamaCppBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody, err := json.Marshal(llamaCppBatchRequest{Content: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		e.endpoint+"/embedding", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llama.cpp batch request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse batch response - array of embedding results
	var batchResult []llamaCppResponseItem
	if err := json.Unmarshal(body, &batchResult); err != nil {
		return nil, fmt.Errorf("failed to parse batch response: %w", err)
	}

	if len(batchResult) != len(texts) {
		return nil, fmt.Errorf("batch response count mismatch: got %d, expected %d", len(batchResult), len(texts))
	}

	results := make([][]float32, len(texts))
	for _, item := range batchResult {
		if item.Index >= len(texts) || len(item.Embedding) == 0 {
			return nil, fmt.Errorf("invalid batch response item: index=%d", item.Index)
		}
		results[item.Index] = item.Embedding[0]
	}

	return results, nil
}

// embedBatchFallback uses concurrent individual requests as fallback
func (e *Embedder) embedBatchFallback(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	// Semaphore for concurrency control
	sem := make(chan struct{}, 8)

	for i, text := range texts {
		wg.Add(1)
		go func(idx int, t string) {
			defer wg.Done()

			sem <- struct{}{}
			defer func() { <-sem }()

			emb, err := e.Embed(ctx, t)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}

			mu.Lock()
			results[idx] = emb
			mu.Unlock()
		}(i, text)
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return results, nil
}

type llamaCppRequest struct {
	Content string `json:"content"`
}

// llamaCppBatchRequest supports sending multiple texts in one request
type llamaCppBatchRequest struct {
	Content []string `json:"content"`
}

// llamaCppResponse handles both formats:
// - Array format: [{"index": 0, "embedding": [[...]]}]
// - Object format: {"embedding": [...]}
type llamaCppResponseItem struct {
	Index     int         `json:"index"`
	Embedding [][]float32 `json:"embedding"` // Nested array in new format
}

func (e *Embedder) callLlamaCpp(ctx context.Context, text string) ([]float32, error) {
	reqBody, err := json.Marshal(llamaCppRequest{Content: text})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		e.endpoint+"/embedding", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llama.cpp request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, string(body))
	}

	// Read raw response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Try array format first (new llama.cpp format)
	var arrayResult []llamaCppResponseItem
	if err := json.Unmarshal(body, &arrayResult); err == nil && len(arrayResult) > 0 {
		if len(arrayResult[0].Embedding) > 0 {
			return arrayResult[0].Embedding[0], nil
		}
	}

	// Try simple object format
	var objectResult struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(body, &objectResult); err == nil && len(objectResult.Embedding) > 0 {
		return objectResult.Embedding, nil
	}

	return nil, fmt.Errorf("failed to parse embedding response: %s", string(body[:min(100, len(body))]))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Stats returns embedder statistics.
func (e *Embedder) Stats() (total, hits, errors int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.totalRequests, e.cacheHits, e.errors
}

// Cache is a simple LRU-ish cache for embeddings.
type Cache struct {
	mu       sync.RWMutex
	data     map[string][]float32
	maxSize  int
	hitCount atomic.Int64
}

func NewCache(maxSize int) *Cache {
	return &Cache{
		data:    make(map[string][]float32),
		maxSize: maxSize,
	}
}

func (c *Cache) Get(key string) []float32 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if v, ok := c.data[key]; ok {
		c.hitCount.Add(1)
		return v
	}
	return nil
}

func (c *Cache) Set(key string, value []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simple eviction: clear half when full
	if len(c.data) >= c.maxSize {
		count := 0
		for k := range c.data {
			delete(c.data, k)
			count++
			if count >= c.maxSize/2 {
				break
			}
		}
	}

	c.data[key] = value
}
