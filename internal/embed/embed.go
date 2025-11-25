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
	"time"
)

const (
	defaultEndpoint  = "http://localhost:8080"
	defaultTimeout   = 30 * time.Second
	maxContextTokens = 1500 // Safe limit for 2048 context (llama.cpp tokens)
)

// Embedder generates embeddings via llama.cpp server.
type Embedder struct {
	endpoint string
	client   *http.Client
	cache    *Cache
	mu       sync.Mutex

	// Stats
	totalRequests int64
	cacheHits     int64
	errors        int64
}

// New creates a new embedder.
func New() *Embedder {
	endpoint := os.Getenv("SGREP_ENDPOINT")
	if endpoint == "" {
		endpoint = defaultEndpoint
	}

	return &Embedder{
		endpoint: endpoint,
		client: &http.Client{
			Timeout: defaultTimeout,
		},
		cache: NewCache(10000),
	}
}

// Embed generates an embedding for the given text.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
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

// EmbedBatch generates embeddings for multiple texts.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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
	defer resp.Body.Close()

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
	hitCount int64
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
		c.hitCount++
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
