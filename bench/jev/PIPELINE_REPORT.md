# No-model-replacement pipeline experiment

## Result

The useful change is **implementation-first ordering**, not unconditional
reranked-order preservation. On the frozen implementation-search labels, a
stable partition that places source files before Markdown, docs, examples, and
tests improved existing Nomic balanced retrieval at effectively zero added
latency. Combining the same policy with the existing local Jina scores produced
the strongest local result in this experiment.

These are fixed-label exploratory results. The ten held-out cases have not had
the requested independent blind-human audit, and all eight positive judgments
target implementation files. This evidence supports an explicit
implementation-search mode; it does **not** justify globally demoting docs for
documentation or conceptual queries.

## Method

- Existing deterministic chunks and frozen `dspy-go` corpus commit
  `87cb50fa0611ef8a3905494c8b16fb0aab7666d3`.
- Existing Nomic dense vectors, FTS5/BM25, 0.6/0.4 weighted RRF, filters, overlap
  deduplication, paths, and line ranges.
- Existing Jina v2 Q8 model through a verified loopback-only llama.cpp endpoint.
- Nine variants declared before the final three-run measurement; no model or
  Jev judged the results.
- Three runs over ten held-out cases. Every ranking was identical across runs.

## Aggregate results

| Arm | MRR | NDCG@10 | R@10 | Prose errors | p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Balanced baseline | 0.667 | 0.499 | 0.438 | 1 | 33 / 56 ms |
| **Balanced implementation-first** | **0.729** | **0.567** | **0.500** | **0** | **33 / 56 ms** |
| Current verified Jina boundary | 0.667 | 0.503 | 0.438 | 1 | 3,602 / 5,150 ms |
| Preserve Jina fused order | 0.604 | 0.506 | 0.438 | 3 | 3,602 / 5,150 ms |
| Post-filter fill only | 0.667 | 0.503 | 0.438 | 1 | 3,602 / 5,150 ms |
| Preserve order + post-filter fill | 0.604 | 0.506 | 0.438 | 3 | 3,602 / 5,150 ms |
| Prefilter shortlist + preserve/fill | 0.604 | 0.506 | 0.438 | 3 | 3,272 / 5,529 ms |
| Preserve/fill + max two chunks/file | 0.629 | 0.544 | 0.500 | 5 | 3,602 / 5,150 ms |
| **Jina preserve/fill + implementation-first** | **0.792** | **0.621** | **0.500** | **0** | 3,602 / 5,150 ms |

Implementation-first Nomic had two MRR wins and no losses versus baseline:
tool registration moved from rank 3 to rank 2, and multimodal implementation
entered at rank 3 instead of being absent. Jina plus implementation-first moved
tool registration to rank 1 and improved the Anthropic ordering to ideal NDCG,
again with no per-case MRR loss. Self-refine remained unretrieved, showing that
ordering cannot repair missing candidate recall.

Both no-answer controls still returned ten results. None of these deterministic
changes adds calibrated answerability.

## Rejected hypotheses

- **Always preserve fused reranker order:** lower MRR and three prose errors.
  Jina's direct preference for prose makes the current order restoration partly
  protective rather than simply a bug.
- **Filtering/deduplication before the shortlist:** no quality improvement and
  the same preserve-order regression.
- **Fill after filtering alone:** no change because every frozen case already
  produced ten results.
- **Two chunks per file:** better NDCG/recall but five prose errors; reject.
- **Identifier-aware FTS:** not scored because the frozen set has no labeled
  identifier-heavy cases. Claiming a benefit from this dataset would be invalid.

## Recommendation

1. **Experiment:** use the opt-in `--implementation-first` mode, which applies
   the source-first stable partition after candidate generation. Do not make it
   a global default until docs-seeking and conceptual queries are independently
   labeled and tested.
2. **Defer:** Jina plus implementation-first has the best local score, but its
   3.6-second CPU median is not an interactive default. The zero-cost Nomic-only
   variant captures most of the measured gain.
3. **Reject:** unconditional reranked-order preservation, prefiltering as a
   quality fix, and the tested file-diversification rule.

## Existing 20-query dspy-go regression benchmark

The same predeclared variants were subsequently run unchanged on the
repository's existing `bench/quality/dspy-go-dataset.json`: 20 shallow graded
queries, 56 implementation-file judgments, and no prose-positive judgments.
This includes the eight positive Jev held-out queries, so the all-20 result is
not independent confirmation. The 12 previously unevaluated queries are
reported separately below.

| Arm | MRR | NDCG@10 | R@10 | Prose errors | p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Balanced baseline | 0.717 | 0.534 | 0.421 | 2 | 33 / 59 ms |
| **Balanced implementation-first** | **0.767** | **0.580** | **0.446** | **0** | **33 / 59 ms** |
| Current verified Jina boundary | 0.742 | 0.548 | 0.404 | 1 | 3,526 / 5,229 ms |
| Preserve Jina fused order | 0.717 | 0.549 | 0.404 | 3 | 3,526 / 5,229 ms |
| Two chunks per file | 0.727 | 0.582 | **0.488** | 5 | 3,526 / 5,229 ms |
| **Jina preserve/fill + implementation-first** | **0.792** | **0.601** | 0.446 | **0** | 3,526 / 5,229 ms |

On the 12 queries not present in the Jev held-out positives, implementation-
first still improved MRR from 0.750 to 0.792 and NDCG@10 from 0.558 to 0.588,
removed the one prose error, and had no recall change (0.410). Jina plus the
policy had the same MRR/recall and slightly lower NDCG (0.587) on those 12,
providing no incremental evidence for paying its latency.

Across the benchmark's four fixed five-query folds, Nomic implementation-first
had no aggregate MRR, NDCG, or recall regression. MRR was tied on folds 0 and 3
and improved on folds 1 and 2; prose errors changed from two to zero. At the
case level, it changed five of 20 queries: tool registration, signature fields,
few-shot bootstrap, multimodal handling, and local Ollama. Each changed case
improved at least one metric and none regressed.

The compiled `sgrep --implementation-first` CLI independently reproduced the
20-query 0.767/0.580/0.446 result. Sequential subprocess medians were 100.4ms
for the unchanged default and 101.2ms with the option; this 0.8ms difference is
within run noise and is not claimed as measured overhead.

The repository's other internal dataset targets a separate `maestro` corpus
that is not present in this checkout, while the CoIR adapter materializes
code-only corpora. The latter cannot test source-versus-prose ordering because
every candidate has a source extension; the policy is a no-op there. Neither
was presented as additional evidence.

## Failure and privacy checks

The live run had zero reranker failures. A separate ten-query bad-endpoint run
produced ten failures and every Jina-dependent arm exactly matched the balanced
baseline, verifying fail-open behavior in the scaffold. Inference was local;
no query or source content left the orb. The JSON artifact stores queries,
paths, ranges, metrics, and timings but no source bodies or credentials.
