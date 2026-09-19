package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"time"
)

const (
	jevModel          = "jev-1.13.0"
	jevEndpoint       = "https://api.typesafe.ai/v1/systemone"
	jevPricePerMTok   = 0.042
	jevShortlistLimit = 10
)

type jevQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

type jevRequest struct {
	State     any                    `json:"state"`
	Model     string                 `json:"model"`
	Questions map[string]jevQuestion `json:"questions"`
}

type jevAnswer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
}

type jevResponse struct {
	Model   string               `json:"model"`
	Answers map[string]jevAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type candidateJudgment struct {
	Index            int                `json:"index"`
	Score            float64            `json:"score"`
	DirectAnswer     float64            `json:"direct_answer"`
	EvidenceType     string             `json:"evidence_type"`
	EvidenceProbs    map[string]float64 `json:"evidence_probabilities"`
	AbstractionLayer string             `json:"abstraction_layer"`
	AbstractionProbs map[string]float64 `json:"abstraction_probabilities"`
	TestOrDocsOnly   float64            `json:"test_or_docs_only"`
	AnswersQuery     float64            `json:"answers_query"`
	InputTokens      int                `json:"input_tokens"`
	OutputTokens     int                `json:"output_tokens"`
	ResponseModel    string             `json:"response_model"`
	RequestLatencyMS float64            `json:"request_latency_ms"`
}

type jevRunStats struct {
	Candidates      []candidateJudgment `json:"candidates,omitempty"`
	InputTokens     int                 `json:"input_tokens"`
	OutputTokens    int                 `json:"output_tokens"`
	Calls           int                 `json:"calls"`
	Failure         string              `json:"failure,omitempty"`
	Fallback        bool                `json:"fallback"`
	AnyAnswerScore  float64             `json:"any_answer_score"`
	RerankLatencyMS float64             `json:"rerank_latency_ms"`
	ResponseModel   string              `json:"response_model,omitempty"`
}

type jevProxy struct {
	apiKey     string
	scoreScale float64
	client     *http.Client
	server     *httptest.Server

	mu       sync.Mutex
	stats    map[string]jevRunStats
	failNext bool
}

func newJevProxy(apiKey string, scoreScale float64) *jevProxy {
	p := &jevProxy{
		apiKey:     apiKey,
		scoreScale: scoreScale,
		client:     &http.Client{Timeout: 45 * time.Second},
		stats:      make(map[string]jevRunStats),
	}
	p.server = httptest.NewServer(http.HandlerFunc(p.handleRerank))
	return p
}

func (p *jevProxy) close() { p.server.Close() }

func (p *jevProxy) endpoint() string { return p.server.URL }

func (p *jevProxy) failOnce() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.failNext = true
}

func (p *jevProxy) takeStats(query string) jevRunStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	stats := p.stats[query]
	delete(p.stats, query)
	return stats
}

func (p *jevProxy) handleRerank(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	var req struct {
		Query     string   `json:"query"`
		Documents []string `json:"documents"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid rerank request", http.StatusBadRequest)
		return
	}

	p.mu.Lock()
	fail := p.failNext
	p.failNext = false
	p.mu.Unlock()
	if fail {
		stats := jevRunStats{Failure: "injected service failure", Fallback: true}
		p.recordStats(req.Query, stats, started)
		http.Error(w, "injected service failure", http.StatusBadGateway)
		return
	}

	limit := min(len(req.Documents), jevShortlistLimit)
	judgments := make([]candidateJudgment, limit)
	errByIndex := make([]error, limit)
	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup
	for i := range limit {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			judgments[i], errByIndex[i] = p.scoreCandidate(r.Context(), req.Query, req.Documents[i], i)
		}()
	}
	wg.Wait()

	stats := jevRunStats{Candidates: judgments, Calls: limit}
	for i, err := range errByIndex {
		if err != nil {
			stats.Failure = fmt.Sprintf("candidate %d: %v", i, err)
			stats.Fallback = true
			p.recordStats(req.Query, stats, started)
			http.Error(w, "Jev request failed; preserving pre-Jev order", http.StatusBadGateway)
			return
		}
		stats.InputTokens += judgments[i].InputTokens
		stats.OutputTokens += judgments[i].OutputTokens
		stats.AnyAnswerScore = max(stats.AnyAnswerScore, judgments[i].AnswersQuery)
		stats.ResponseModel = judgments[i].ResponseModel
	}
	p.recordStats(req.Query, stats, started)

	results := make([]map[string]any, limit)
	for i, judgment := range judgments {
		results[i] = map[string]any{
			"index":           judgment.Index,
			"relevance_score": probabilityLogit(scaledProbability(judgment.Score, p.scoreScale)),
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"model": jevModel, "results": results})
}

func (p *jevProxy) recordStats(query string, stats jevRunStats, started time.Time) {
	stats.RerankLatencyMS = float64(time.Since(started).Microseconds()) / 1000
	p.mu.Lock()
	p.stats[query] = stats
	p.mu.Unlock()
}

func (p *jevProxy) scoreCandidate(ctx context.Context, query, candidate string, index int) (candidateJudgment, error) {
	request := jevRequest{
		State: map[string]any{
			"query": query,
			"candidate": map[string]string{
				"content": candidate,
			},
		},
		Model: jevModel,
		Questions: map[string]jevQuestion{
			"direct_answer": {
				Type:         "noul",
				Instructions: "Does `candidate.content` directly provide implementation evidence that answers the code-search intent in `query`?",
				Criteria: map[string]string{
					"true":  "The candidate contains the implementation, API definition, call path, or configuration logic needed to answer the query.",
					"false": "The candidate only shares words, gives broad prose, shows usage without the implementation, or concerns a different behavior.",
				},
			},
			"evidence_type": {
				Type:         "choice",
				Instructions: "What kind of evidence does `candidate.content` provide for `query`? Choose by the candidate's actual role, not by shared terminology.",
				Criteria: map[string]string{
					"implementation": "Production implementation, defining API, or direct call path for the requested behavior.",
					"test_or_docs":   "Only a test, example, README, guide, or descriptive comment about the behavior.",
					"incidental":     "Unrelated or merely an incidental terminology match.",
				},
			},
			"abstraction_layer": {
				Type:         "choice",
				Instructions: "Is `candidate.content` at the abstraction layer requested by `query`?",
				Criteria: map[string]string{
					"correct":  "It implements or defines the requested behavior at the layer a developer would inspect to understand or change it.",
					"adjacent": "It is a caller, wrapper, supporting type, test, or example one layer away.",
					"wrong":    "It is about a different subsystem or only happens to use similar words.",
				},
			},
			"test_or_docs_only": {
				Type:         "noul",
				Instructions: "Is `candidate.content` only test, example, or documentation evidence rather than the production implementation requested by `query`?",
				Criteria: map[string]string{
					"true":  "It demonstrates or describes behavior but does not contain the production implementation or defining API.",
					"false": "It contains production implementation or a defining production API, even if it also has comments.",
				},
			},
			"answers_query": {
				Type:         "noul",
				Instructions: "Could a developer use `candidate.content` to answer the concrete code-search question in `query` without relying on mere keyword overlap?",
				Criteria: map[string]string{
					"true":  "It contains directly useful evidence for answering the query, whether primary implementation or essential supporting code.",
					"false": "It does not answer the query; topical similarity, generic prose, and incidental mentions count as false.",
				},
			},
		},
	}

	started := time.Now()
	response, err := p.call(ctx, request)
	if err != nil {
		return candidateJudgment{}, err
	}
	answer := func(name, expected string) (jevAnswer, error) {
		a, ok := response.Answers[name]
		if !ok || a.Type != expected {
			return jevAnswer{}, fmt.Errorf("answer %q has type %q, want %q", name, a.Type, expected)
		}
		return a, nil
	}
	direct, err := answer("direct_answer", "noul")
	if err != nil {
		return candidateJudgment{}, err
	}
	evidence, err := answer("evidence_type", "choice")
	if err != nil {
		return candidateJudgment{}, err
	}
	layer, err := answer("abstraction_layer", "choice")
	if err != nil {
		return candidateJudgment{}, err
	}
	testDocs, err := answer("test_or_docs_only", "noul")
	if err != nil {
		return candidateJudgment{}, err
	}
	answersQuery, err := answer("answers_query", "noul")
	if err != nil {
		return candidateJudgment{}, err
	}

	// Jev supplies semantic judgments; deterministic code owns arithmetic.
	score := 0.40*direct.Noul +
		0.20*evidence.Probabilities["implementation"] +
		0.15*layer.Probabilities["correct"] +
		0.25*answersQuery.Noul -
		0.15*testDocs.Noul -
		0.10*evidence.Probabilities["incidental"]
	score = max(0, min(1, score))

	return candidateJudgment{
		Index:            index,
		Score:            score,
		DirectAnswer:     direct.Noul,
		EvidenceType:     evidence.Choice,
		EvidenceProbs:    evidence.Probabilities,
		AbstractionLayer: layer.Choice,
		AbstractionProbs: layer.Probabilities,
		TestOrDocsOnly:   testDocs.Noul,
		AnswersQuery:     answersQuery.Noul,
		InputTokens:      response.Usage.InputTokens,
		OutputTokens:     response.Usage.OutputTokens,
		ResponseModel:    response.Model,
		RequestLatencyMS: float64(time.Since(started).Microseconds()) / 1000,
	}, nil
}

func (p *jevProxy) call(ctx context.Context, payload jevRequest) (jevResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return jevResponse{}, fmt.Errorf("marshal request: %w", err)
	}
	for attempt := range 3 {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, jevEndpoint, bytes.NewReader(body))
		if err != nil {
			return jevResponse{}, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := p.client.Do(req)
		if err != nil {
			return jevResponse{}, fmt.Errorf("request: %w", err)
		}
		responseBody, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		_ = resp.Body.Close()
		if readErr != nil {
			return jevResponse{}, fmt.Errorf("read response: %w", readErr)
		}
		if resp.StatusCode == http.StatusOK {
			var result jevResponse
			if err := json.Unmarshal(responseBody, &result); err != nil {
				return jevResponse{}, fmt.Errorf("decode response: %w", err)
			}
			if result.Model != jevModel {
				return jevResponse{}, fmt.Errorf("response model %q, want pinned %q", result.Model, jevModel)
			}
			return result, nil
		}
		if (resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode != 529) || attempt == 2 {
			return jevResponse{}, fmt.Errorf("status %d: %s", resp.StatusCode, sanitizeAPIError(responseBody))
		}
		delay := time.Duration(1<<attempt) * 250 * time.Millisecond
		if seconds, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && seconds > 0 {
			delay = time.Duration(seconds) * time.Second
		}
		select {
		case <-ctx.Done():
			return jevResponse{}, ctx.Err()
		case <-time.After(delay):
		}
	}
	return jevResponse{}, errors.New("request retry loop exhausted")
}

func sanitizeAPIError(body []byte) string {
	var response struct {
		Detail any `json:"detail"`
	}
	if json.Unmarshal(body, &response) == nil && response.Detail != nil {
		return fmt.Sprintf("%v", response.Detail)
	}
	return http.StatusText(http.StatusBadGateway)
}

func probabilityLogit(p float64) float64 {
	p = max(0.001, min(0.999, p))
	return math.Log(p / (1 - p))
}

func scaledProbability(p, scale float64) float64 {
	return 0.5 + scale*(p-0.5)
}
