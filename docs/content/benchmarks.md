---
title: Benchmarks
description: Final pooled cross-tool retrieval results plus separately bounded internal index and vector measurements.
weight: 40
---

## Final pooled local verdict

The final cross-tool harness compared sgrep, grepai, osgrep, and ChunkHound on
an Apple M3 Pro using 20 queries against `dspy-go` commit `87cb50f`. The tested
sgrep source was `19ebec1`. Rankings were frozen before the pooled qrels were
finalized, then rescored without rerunning the tools.

`sgrep balanced` is the strongest overall local quality-latency tradeoff and
the strongest normalized implementation-code retriever by point estimate. It
is not universally best. The 95% query-bootstrap confidence intervals overlap,
so the ordering is not a claim of statistical significance.

### Normalized implementation-code results

The normalized track requests 50 candidates, applies a common
implementation-file scope, deduplicates by first file occurrence, and scores
up to ten unique files.

| Tool/profile | MRR | NDCG@10 | R@10 | Median latency |
|--------------|----:|--------:|-----:|---------------:|
| `sgrep fast` | 0.749 | 0.532 | 0.377 | **42.0ms** |
| `sgrep balanced` | **0.792** | **0.589** | **0.413** | 54.4ms |
| `sgrep quality` | **0.792** | 0.555 | 0.372 | 69.3ms |
| grepai | 0.585 | 0.436 | 0.332 | 110.2ms |
| osgrep | 0.733 | 0.462 | 0.292 | 939.4ms |
| ChunkHound | 0.679 | 0.493 | 0.396 | 2946.0ms |

Balanced is recommended over quality: it is faster, ties its MRR, and has
higher NDCG@10 and R@10 point estimates in this track.

### Other local evaluation tracks

| Track | Point-estimate leaders |
|-------|------------------------|
| Normalized all files | osgrep MRR **0.867**; sgrep balanced/quality NDCG@10 **0.607** and R@10 **0.340** |
| Native product output | osgrep MRR **0.858**; ChunkHound R@10 **0.236**; sgrep balanced NDCG@10 **0.429** in the full pool |

The product track preserves each CLI's native order and duplicate file hits.
Its small NDCG@10 ordering is pool-sensitive: when candidates exclusive to the
sgrep family are left out, ChunkHound is 0.426, sgrep fast is 0.421, and sgrep
balanced is 0.413. This is why the full-pool product NDCG point estimate is not
presented as a robust win. The normalized local leaders do not change under
that sensitivity check.

### Judgment and cloud caveats

The pooled qrels are model-judged, not human-annotated. Two independent full
passes were followed by blind adjudication. The reported 95% percentile
intervals use 10,000 query bootstrap samples; with only 20 queries, they are
wide and overlap. Product defaults also differ in model, chunking, and storage,
so the benchmark measures whole retrieval products rather than an isolated
algorithm.

Cloud-backed mgrep remains in a separate exploratory track. Its run was
quota-interrupted and its timing is a completion-interval proxy, not official
CLI wall time. It is therefore excluded from the local quality-latency verdict.

## Separate internal index and vector measurements

The measurements below come from an earlier sgrep-only after-state snapshot on
the same pinned corpus. They do not share the final cross-tool harness's
executable, pooled judgments, or external CLI timing boundary. In particular,
internal search-stage times exclude CLI process startup, while the final table
above reports external CLI wall time.

In that internal snapshot, mean stage timings were 22.6ms for query embedding
in the quality profile, 12.1ms for its vector stage, and 11.7ms for ColBERT.
The fast profile's vector stage measured 4.3ms. Do not add these stages to, or
compare them directly with, the final cross-tool medians.

### Index time and storage

The base index took 3m40.95s. Full indexing with the default ColBERT precompute and export took 7m42.59s.

| Artifact | Size |
|----------|-----:|
| SQLite index | 62MB |
| Exact float32 mmap | 39MB |
| ColBERT mmap | 23MB |
| TQ-MSE chunk vectors | 5.3MB |
| File vectors | 222KB |
| **Total** | **about 130MB** |

### Vector microbenchmarks

Five-run top-50 scan measurements:

| Backend | 10k vectors | 100k vectors | Storage tradeoff |
|---------|------------:|-------------:|------------------|
| Exact float32 | 0.63ms | 6.8ms | Largest resident artifact |
| TQ-MSE | 4.9ms | 49.3ms | About 7.4× smaller vector storage |
| Binary coarse scan | 0.43ms | 4.3ms | Intended as a coarse filter before reranking |

At the real 13,136-chunk corpus, the exact first-stage vector time was 4.3ms versus 7.1ms with TQ-MSE forced. The production cutoff remains 20,000 chunks to cap exact artifact memory and disk use.

## Methodology limits

- Final retrieval scores use 20 model-judged queries; overlapping bootstrap
  intervals preclude a strong general or significance claim.
- The product/native NDCG ordering changes under sgrep-family pool leaveout;
  normalized point-estimate leaders remain unchanged in that check.
- Product defaults use different models, chunking, and storage engines, so
  they do not isolate one component.
- The index, storage, stage-timing, and vector measurements are an internal
  after-state snapshot, not a controlled same-index pre/post A/B test and not
  the final cross-tool run boundary.
- Latency depends on corpus size, hardware, warm state, profile, vector
  artifact, and whether process startup is included.

The committed internal-snapshot query logs are in
[`bench/results/current-implementation`](https://github.com/XiaoConstantine/sgrep/tree/main/bench/results/current-implementation).
They reproduce the internal smoke boundary, not the final pooled cross-tool
scores. Reproduction should pin source and corpus commits, use full
repository-relative qrel paths, fail on subprocess errors and timeouts, and
report warmup policy, concurrency, and whether latency includes process
startup.
