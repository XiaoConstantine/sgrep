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

### Result policy

Historical latency numbers were retired because retrieval modes could be silently contaminated by an existing ColBERT artifact and six subprocesses contended for one embedding server. New runs:

- force `fast`, `balanced`, and `quality` profiles;
- execute sequentially by default and report median/p95 latency;
- use `--concurrency N` only for separate throughput tests;
- preserve corpus-relative paths, require unambiguous judgments, and verify the pinned corpus commit;
- record concurrency and corpus metadata in JSON summaries.

Cross-encoder reranking remains experimental and off by default because general-text rerankers have historically demoted code-relevant results.

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

# Quality profile only
uv run bench/quality/run_dspy_bench.py --tool sgrep --mode hybrid+colbert

# Tune weighted RRF on one deterministic four-fold training split
uv run bench/quality/run_dspy_bench.py --tool sgrep --mode hybrid --split train --fold 0 --semantic-weight 0.6 --bm25-weight 0.4

# Confirm the selected weights on that five-query held-out fold
uv run bench/quality/run_dspy_bench.py --tool sgrep --mode hybrid --split test --fold 0 --semantic-weight 0.6 --bm25-weight 0.4

# Repeat with --fold 1, 2, and 3 to aggregate all held-out queries.

# Separate six-client throughput/contention run
uv run bench/quality/run_dspy_bench.py --tool sgrep --mode all --concurrency 6

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
