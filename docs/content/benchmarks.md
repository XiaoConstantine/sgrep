---
title: Benchmarks
description: Current retrieval-quality, latency, indexing, and vector-scan measurements with their limitations.
weight: 40
---

## Current profile comparison

The current after-state benchmark indexes `dspy-go` commit `87cb50fa0611ef8a3905494c8b16fb0aab7666d3`: 543 files, 13,136 chunks, and 61,082 ColBERT segments. It uses Nomic Q8 with a 512-token context on an Apple M3 Pro with 20 judged queries, top-10 retrieval, and sequential concurrency of one.

| Profile | Pipeline | MRR | Recall@5 | Mean | Median | p95 |
|---------|----------|----:|---------:|-----:|-------:|----:|
| `fast` | Semantic | 0.470 | 0.300 | 30.2ms | 29ms | 37ms |
| `balanced` | Semantic + BM25 | 0.593 | 0.350 | 35.9ms | 36ms | 44ms |
| `quality` | Hybrid + ColBERT | **0.729** | **0.408** | 46.5ms | 46.5ms | 56ms |

Balanced improves absolute MRR by 0.123 over fast for 5.7ms additional mean latency in this run. Quality adds another 0.136 absolute MRR for 10.6ms over balanced. These differences describe this corpus and judgment set; they are not universal latency or quality guarantees.

Mean stage timings were 22.6ms for query embedding in the quality profile, 12.1ms for its vector stage, and 11.7ms for ColBERT. The fast profile's vector stage measured 4.3ms.

## Index time and storage

The base index took 3m40.95s. Full indexing with the default ColBERT precompute and export took 7m42.59s.

| Artifact | Size |
|----------|-----:|
| SQLite index | 62MB |
| Exact float32 mmap | 39MB |
| ColBERT mmap | 23MB |
| TQ-MSE chunk vectors | 5.3MB |
| File vectors | 222KB |
| **Total** | **about 130MB** |

## Vector microbenchmarks

Five-run top-50 scan measurements:

| Backend | 10k vectors | 100k vectors | Storage tradeoff |
|---------|------------:|-------------:|------------------|
| Exact float32 | 0.63ms | 6.8ms | Largest resident artifact |
| TQ-MSE | 4.9ms | 49.3ms | About 7.4× smaller vector storage |
| Binary coarse scan | 0.43ms | 4.3ms | Intended as a coarse filter before reranking |

At the real 13,136-chunk corpus, the exact first-stage vector time was 4.3ms versus 7.1ms with TQ-MSE forced. The production cutoff remains 20,000 chunks to cap exact artifact memory and disk use.

## Methodology limits

- This is an after-state profile comparison, not a controlled same-index pre/post A/B test.
- Twenty judged queries are too few for a strong general quality claim.
- The four held-out folds and 256/512-token context sweep have not yet been run.
- The cross-encoder cascade was not measured because its separate reranker model was absent.
- Historical README MRR was 0.725; the current 0.729 is not evidence of a material improvement over that prior run.
- Latency depends on corpus size, hardware, warm state, profile, and vector artifact. Compare sequential p50/p95 separately from concurrent throughput.

The committed query logs are in [`bench/results/current-implementation`](https://github.com/XiaoConstantine/sgrep/tree/main/bench/results/current-implementation). Reproduction should pin the corpus commit, use full repository-relative qrel paths, fail on subprocess errors and timeouts, and report warmup policy and concurrency.
