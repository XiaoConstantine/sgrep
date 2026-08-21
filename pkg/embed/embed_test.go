package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
	"github.com/XiaoConstantine/sgrep/pkg/util"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Endpoint != "" {
		t.Errorf("expected managed mode with empty endpoint, got %s", cfg.Endpoint)
	}
	if cfg.Timeout != defaultTimeout {
		t.Errorf("expected timeout %v, got %v", defaultTimeout, cfg.Timeout)
	}
	if cfg.CacheSize != 10000 {
		t.Errorf("expected cache size 10000, got %d", cfg.CacheSize)
	}
	if !cfg.AutoStart {
		t.Error("expected AutoStart to be true")
	}
}

func TestDefaultConfig_CustomEndpoint(t *testing.T) {
	t.Setenv("SGREP_ENDPOINT", "http://custom:9090")

	cfg := DefaultConfig()
	if cfg.Endpoint != "http://custom:9090" {
		t.Errorf("expected custom endpoint, got %s", cfg.Endpoint)
	}
}

func TestNewWithConfig(t *testing.T) {
	cfg := Config{
		Endpoint:  "http://localhost:8888",
		Timeout:   5 * time.Second,
		CacheSize: 500,
		AutoStart: false,
	}

	e := NewWithConfig(cfg)

	if e.endpoint != "http://localhost:8888" {
		t.Errorf("expected endpoint http://localhost:8888, got %s", e.endpoint)
	}
	if e.autoStart {
		t.Error("expected autoStart to be false")
	}
}

func TestNewWithConfig_Defaults(t *testing.T) {
	cfg := Config{}

	e := NewWithConfig(cfg)

	if e.endpoint != defaultEndpoint {
		t.Errorf("expected default endpoint, got %s", e.endpoint)
	}
	if e.client.Timeout != defaultTimeout {
		t.Errorf("expected default timeout, got %v", e.client.Timeout)
	}
}

func TestNew_DefaultManagedEndpointHonorsSGREPPort(t *testing.T) {
	t.Setenv("SGREP_HOME", t.TempDir())
	t.Setenv("SGREP_ENDPOINT", "")
	t.Setenv("SGREP_PORT", "19090")
	e := New()
	if e.serverMgr == nil {
		t.Fatal("expected managed server")
	}
	if e.endpoint != "http://localhost:19090" {
		t.Fatalf("endpoint = %q, want managed SGREP_PORT endpoint", e.endpoint)
	}
}

func TestNewWithConfig_CustomEndpointWithAutoStart(t *testing.T) {
	cfg := Config{
		Endpoint:  "http://localhost:8091",
		AutoStart: true,
	}

	e := NewWithConfig(cfg)

	if e.endpoint != "http://localhost:8091" {
		t.Errorf("expected explicit endpoint to be preserved, got %s", e.endpoint)
	}
	if e.serverMgr != nil {
		t.Error("expected default server manager to be disabled for explicit endpoint")
	}
}

func TestNew(t *testing.T) {
	t.Setenv("SGREP_HOME", t.TempDir())
	t.Setenv("SGREP_ENDPOINT", "")
	e := New()

	if e == nil {
		t.Fatal("New should return non-nil embedder")
	}
	if e.cache == nil {
		t.Error("cache should be initialized")
	}
	if e.serverMgr == nil {
		t.Error("default embedder should use managed server mode")
	}
}

func TestNewWithOptions(t *testing.T) {
	e := NewWithOptions(false)

	if e.autoStart {
		t.Error("expected autoStart to be false")
	}
}

func TestEmbedder_Embed_MockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embedding" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != "POST" {
			t.Errorf("unexpected method: %s", r.Method)
		}

		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(i) * 0.001
		}

		resp := struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	vec, err := e.Embed(context.Background(), "test text")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(vec) != 768 {
		t.Errorf("expected 768 dimensions, got %d", len(vec))
	}
}

func TestEmbedder_Embed_ArrayFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(i) * 0.001
		}

		resp := []struct {
			Index     int         `json:"index"`
			Embedding [][]float32 `json:"embedding"`
		}{
			{Index: 0, Embedding: [][]float32{embedding}},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	vec, err := e.Embed(context.Background(), "test text")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(vec) != 768 {
		t.Errorf("expected 768 dimensions, got %d", len(vec))
	}
}

func TestEmbedder_Embed_Caching(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		embedding := make([]float32, 768)
		resp := struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	_, _ = e.Embed(context.Background(), "same text")
	_, _ = e.Embed(context.Background(), "same text")
	_, _ = e.Embed(context.Background(), "same text")

	if callCount != 1 {
		t.Errorf("expected 1 server call due to caching, got %d", callCount)
	}
}

func TestEmbedder_Embed_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("error message"))
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected error for HTTP 400")
	}
}

func TestEmbedder_Embed_InvalidResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected error for invalid response")
	}
}

func TestEmbedder_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedding := make([]float32, 768)
		resp := struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	texts := []string{"text1", "text2", "text3"}
	results, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}

	for i, r := range results {
		if len(r) != 768 {
			t.Errorf("result[%d] has wrong length: %d", i, len(r))
		}
	}
}

func TestEmbedder_EmbedBatch_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	texts := []string{"text1", "text2"}
	_, err := e.EmbedBatch(context.Background(), texts)
	if err == nil {
		t.Error("expected error from EmbedBatch")
	}
}

func TestEmbedder_Stats(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedding := make([]float32, 768)
		resp := struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	_, _ = e.Embed(context.Background(), "text1")
	_, _ = e.Embed(context.Background(), "text1") // cache hit
	_, _ = e.Embed(context.Background(), "text2")

	total, hits, errors := e.Stats()

	if total != 3 {
		t.Errorf("expected total 3, got %d", total)
	}
	if hits != 1 {
		t.Errorf("expected hits 1, got %d", hits)
	}
	if errors != 0 {
		t.Errorf("expected errors 0, got %d", errors)
	}
}

func TestEmbedder_EventBox(t *testing.T) {
	eb := util.NewEventBox()

	e := NewWithConfig(Config{
		Endpoint:  "http://localhost:19999",
		AutoStart: false,
		EventBox:  eb,
	})

	if e.eventBox != eb {
		t.Error("eventBox should be set")
	}
}

func TestEmbedRejectsOversizedInputWithoutTruncation(t *testing.T) {
	t.Setenv("SGREP_CONTEXT_TOKENS", "128")
	e := NewWithConfig(Config{Endpoint: "http://127.0.0.1:1", AutoStart: false})
	text := strings.Repeat("word ", 1000)
	if _, err := e.Embed(context.Background(), text); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("Embed oversized input error = %v, want size error", err)
	}
}

func TestRetrievalTaskPrefixes(t *testing.T) {
	if got := modelcfg.QueryText("needle"); got != "search_query: needle" {
		t.Fatalf("QueryText = %q", got)
	}
	if got := modelcfg.DocumentText("haystack"); got != "search_document: haystack" {
		t.Fatalf("DocumentText = %q", got)
	}
}

func TestCache(t *testing.T) {
	c := NewCache(100)

	c.Set("key1", []float32{1.0, 2.0, 3.0})
	c.Set("key2", []float32{4.0, 5.0, 6.0})

	v1 := c.Get("key1")
	if v1 == nil {
		t.Error("expected to find key1")
	}
	if len(v1) != 3 {
		t.Errorf("expected 3 elements, got %d", len(v1))
	}

	v2 := c.Get("nonexistent")
	if v2 != nil {
		t.Error("expected nil for nonexistent key")
	}
}

func TestCache_Eviction(t *testing.T) {
	c := NewCache(10)

	for i := 0; i < 20; i++ {
		c.Set("key"+string(rune('a'+i)), []float32{float32(i)})
	}

	count := 0
	for i := 0; i < 26; i++ {
		if c.Get("key"+string(rune('a'+i))) != nil {
			count++
		}
	}

	if count == 0 {
		t.Error("at least some keys should remain after eviction")
	}
}

func TestCache_Concurrent(t *testing.T) {
	c := NewCache(1000)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := "key" + string(rune('a'+i%26))
			c.Set(key, []float32{float32(i)})
		}(i)
	}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := "key" + string(rune('a'+i%26))
			c.Get(key)
		}(i)
	}

	wg.Wait()
}

func TestMin(t *testing.T) {
	if min(5, 10) != 5 {
		t.Error("min(5, 10) should be 5")
	}
	if min(10, 5) != 5 {
		t.Error("min(10, 5) should be 5")
	}
	if min(5, 5) != 5 {
		t.Error("min(5, 5) should be 5")
	}
}

// Benchmarks

func BenchmarkEmbedder_Embed_CacheHit(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedding := make([]float32, 768)
		resp := struct {
			Embedding []float32 `json:"embedding"`
		}{Embedding: embedding}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})

	_, _ = e.Embed(context.Background(), "test text")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = e.Embed(context.Background(), "test text")
	}
}

func BenchmarkEmbedder_CallLlamaCppBatch(b *testing.B) {
	const batchSize = 128

	response := make([]llamaCppResponseItem, batchSize)
	for i := range response {
		embedding := make([]float32, 768)
		for j := range embedding {
			embedding[j] = float32(j) / 768
		}
		response[i] = llamaCppResponseItem{
			Index:     i,
			Embedding: [][]float32{embedding},
		}
	}
	responseBody, err := json.Marshal(response)
	if err != nil {
		b.Fatal(err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(responseBody)
	}))
	b.Cleanup(server.Close)

	e := NewWithConfig(Config{
		Endpoint:  server.URL,
		AutoStart: false,
	})
	texts := make([]string, batchSize)
	for i := range texts {
		texts[i] = "benchmark text"
	}

	b.ReportAllocs()
	b.SetBytes(int64(len(responseBody)))
	b.ResetTimer()
	for b.Loop() {
		results, err := e.callLlamaCppBatch(context.Background(), texts)
		if err != nil {
			b.Fatal(err)
		}
		if len(results) != batchSize {
			b.Fatalf("got %d results, want %d", len(results), batchSize)
		}
	}
}

func BenchmarkCache_Set(b *testing.B) {
	c := NewCache(10000)
	vec := make([]float32, 768)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Set("key", vec)
	}
}

func BenchmarkCache_Get(b *testing.B) {
	c := NewCache(10000)
	vec := make([]float32, 768)
	c.Set("key", vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Get("key")
	}
}

func BenchmarkValidateEmbeddingInput(b *testing.B) {
	text := strings.Repeat("This is a test sentence. ", 50)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = modelcfg.ValidateInput(text)
	}
}
