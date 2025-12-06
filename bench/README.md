# sgrep Benchmarks

This directory contains quality and performance benchmarks for sgrep.

## Directory Structure

```
bench/
├── quality/           # Quality evaluation (IR metrics)
│   ├── types.go       # QueryCase, LabeledResult, Relevance types
│   ├── metrics.go     # MRR, NDCG, MAP, P@k, R@k implementations
│   ├── runner.go      # Evaluation runner
│   ├── dataset.json   # Ground truth evaluation dataset
│   └── *_test.go      # Unit tests
├── results/           # Benchmark output (gitignored)
│   ├── profiles/      # CPU/memory profiles
│   └── *.txt          # Benchmark results
└── search_benchmark_test.go  # End-to-end search benchmarks
```

## Comparison Benchmark (dspy-go corpus)

Compare sgrep against other semantic search tools on the dspy-go codebase (20 queries):

```bash
uv run bench/quality/run_dspy_bench.py --tool all --mode all
```

### Latest Results

| Tool | MRR | P@5 | R@5 | Latency | Tokens | Cost |
|------|-----|-----|-----|---------|--------|------|
| **sgrep (hybrid+colbert)** | **0.698** | 0.230 | 0.425 | 9461ms | 2757 | $0.17 |
| sgrep (hybrid) | 0.615 | 0.200 | 0.358 | 6246ms | 4016 | $0.24 |
| sgrep (semantic) | 0.610 | 0.200 | 0.358 | 5823ms | 2473 | $0.15 |
| sgrep (cascade) | 0.596 | 0.220 | 0.408 | 10564ms | 3121 | $0.19 |
| mgrep (cloud) | 0.262 | 0.050 | 0.100 | 939ms | 4646 | $0.28 |
| osgrep | 0.050 | 0.010 | 0.017 | 916ms | 332 | $0.02 |

**Key findings:**
- **sgrep (hybrid+colbert)** achieves best MRR (0.698) - 2.7x better than mgrep, 14x better than osgrep
- ColBERT late interaction scoring significantly improves accuracy over plain hybrid (+13%)
- Cascade (hybrid+colbert+cross-encoder) hurts performance - cross-encoder demotes code-relevant results
- Cross-encoders like jina-reranker-v2 and mxbai-rerank are trained on general text, not code
- Token count measures actual code content returned (for LLM context estimation)
- Cost estimated at $3/1M tokens (Claude pricing)

### Supported Tools

| Tool | Type | Description |
|------|------|-------------|
| `sgrep` | Local | This tool - semantic + BM25 hybrid search |
| `osgrep` | Local | Open-source semantic search (requires `npm i -g osgrep`) |
| `mgrep` | Cloud | Mixedbread cloud search (requires `npm i -g @mixedbread/mgrep && mgrep login`) |

### Usage

```bash
# Test all sgrep configurations
uv run bench/quality/run_dspy_bench.py --tool sgrep --mode all

# Compare specific configuration
uv run bench/quality/run_dspy_bench.py --tool sgrep --hybrid --colbert

# Test all tools
uv run bench/quality/run_dspy_bench.py --tool all --mode all
```

## Quality Benchmarks

Evaluate search result quality using IR (Information Retrieval) metrics:

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank - position of first relevant result |
| **NDCG@k** | Normalized Discounted Cumulative Gain - graded relevance with position discount |
| **MAP** | Mean Average Precision - overall retrieval quality |
| **P@k** | Precision at k - fraction of relevant in top-k |
| **R@k** | Recall at k - fraction of relevant found in top-k |

### Running Quality Benchmarks

```bash
# Full evaluation against dataset
make bench-quality

# Or directly:
go run ./cmd/sgrep-bench quality -codebase /path/to/repo -dataset bench/quality/dataset.json

# Quick single-query comparison
go run ./cmd/sgrep-bench compare -codebase . -query "how does authentication work"
```

### Ground Truth Dataset

The dataset (`bench/quality/dataset.json`) contains:
- **Queries**: Natural language search intents
- **Judgments**: Labeled relevant files with graded relevance (0/1/2)
- **Categories**: conceptual, api, architecture, edge_case
- **Grep patterns**: Baseline patterns for ripgrep comparison

Example:
```json
{
  "query": "embedding generation",
  "category": "conceptual",
  "judgments": [
    {"file": "embedding_router.go", "rel": 2},
    {"file": "embedding_cache.go", "rel": 2},
    {"file": "embedding_options.go", "rel": 1}
  ],
  "grep_patterns": ["embedding", "Embedding", "embed", "vector"]
}
```

Relevance levels:
- `2` = Highly relevant (primary implementation)
- `1` = Relevant (supporting/related)
- `0` = Not relevant

## Performance Benchmarks

### Running Performance Benchmarks

```bash
# Run all benchmarks
make bench

# Run quick benchmarks (skip large tests)
make bench-quick

# Save baseline for regression detection
make bench-baseline

# Run and compare to baseline
make bench-compare

# Run with CPU/memory profiling
make bench-profile
```

### Benchmark Tests

Located in `internal/bench/` and `bench/`:

- **Vector operations**: L2Distance, batch distance, TopK selection
- **Store operations**: Search at various document counts (1k, 10k, 50k)
- **Chunking**: File parsing and chunking performance
- **End-to-end**: Full search pipeline benchmarks

### Profiling

After running `make bench-profile`:

```bash
# View CPU profile (opens in browser)
go tool pprof -http=:8080 bench/results/profiles/cpu_*.prof

# View memory profile
go tool pprof -http=:8081 bench/results/profiles/mem_*.prof
```

### Regression Detection

Uses [benchstat](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat) for statistical comparison:

```bash
# Install benchstat
go install golang.org/x/perf/cmd/benchstat@latest

# Create baseline on main branch
git checkout main
make bench-baseline

# Switch to feature branch and compare
git checkout feature-branch
make bench-compare
```

## Adding New Test Cases

1. Edit `bench/quality/dataset.json`
2. Add query with relevant file judgments
3. Run `make bench-quality` to evaluate

## Corpus Management

For reproducible benchmarks, pin your test codebase to a specific commit:

```json
{
  "corpus": "maestro",
  "corpus_hash": "abc123..."
}
```

Store corpus snapshots in `bench/corpora/` (gitignored for large repos).
