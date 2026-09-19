# CodeRankEmbed retrieval experiment

## Decision

**Reject CodeRankEmbed as sgrep's default retriever on this evidence.** It did
not beat the current Nomic balanced retriever, did not beat the previously
reported Jina arm, increased implementation-versus-prose errors, doubled query
latency, and required a 46m47s CPU indexing pass for this 9,172-chunk corpus.

The strongest CodeRank result was CodeRank balanced + a verified live Jina
endpoint. It improved recall over Nomic + verified Jina, but remained below the
previously reported Jina MRR/NDCG, added a prose error, and took 4.68s p50.
That is an experiment result, not an adoption case.

All quality scores are fixed-label exploratory evidence. The frozen labels
have not received the requested blind human audit, and no model was used as a
judge.

## Method

- Corpus: `dspy-go` commit `87cb50fa0611ef8a3905494c8b16fb0aab7666d3`.
- 9,172 existing deterministic sgrep chunks and unchanged SQLite FTS5 index.
- Model: `nomic-ai/CodeRankEmbed`, revision
  `3c4b60807d71f79b43f3c4363786d9493691f8b1`, with its required query prefix
  `Represent this query for searching relevant code: `.
- Native 8,192-token model context; actual corpus maximum was 1,213 tokens.
- Exact CPU float32 embeddings, normalized cosine retrieval; weighted RRF
  remained 0.6 dense / 0.4 BM25.
- Existing filters, significant-overlap deduplication, paths, line ranges,
  top-20 Jina boundary, score fusion, and output limit were reconstructed from
  production code. The Nomic control exactly reproduced the frozen baseline.
- Three query runs. Ranking and metrics were identical across runs.

## Results

Eight answerable held-out cases contribute to quality metrics; two frozen
no-answer controls contribute only to answerability observations.

| Arm | MRR | NDCG@10 | R@10 | Candidate R@50 | Prose errors | p50 / p95 end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| Nomic balanced control | **0.667** | **0.499** | 0.438 | **0.646** | 1 | **33.2 / 57.9 ms** |
| Verified Nomic + live Jina | 0.667 | 0.503 | 0.438 | **0.646** | 1 | 3,641 / 5,504 ms |
| CodeRank dense | 0.542 | 0.439 | 0.438 | 0.521 | **0** | 74.6 / 87.7 ms |
| CodeRank balanced | 0.577 | 0.423 | 0.469 | 0.615 | 9 | 74.7 / 87.8 ms |
| CodeRank balanced + live Jina | **0.708** | **0.553** | **0.552** | 0.615 | 2 | 4,681 / 5,867 ms |

Against Nomic balanced, CodeRank balanced had one MRR win, two losses, and five
ties by case. It recovered self-refine and more logging recall, but regressed
ReAct, tool registration, and logging rank quality. The hybrid arm's nine prose
promotion errors versus Nomic's one are the clearest rejection signal for the
stated implementation-versus-prose objective. Dense-only CodeRank avoided
promotion errors but had the lowest candidate recall and lower aggregate
quality.

CodeRank + live Jina recovered self-refine and logging results, producing one
MRR win and no losses against verified Nomic + live Jina. It still did not beat
the old reported Jina figures (0.729/0.565/0.500): MRR and NDCG were lower,
recall was higher, and prose errors doubled.

Neither embedding model exposes a calibrated any-answer judgment. Every arm
returned results for both no-answer controls, so interpreting non-empty output
as “answer found” gives 0/2 no-answer rejections. Do not infer answerability
from cosine score without a separately labeled calibration set.

## Jina control finding

The old “balanced + existing Jina” result (0.729/0.565/0.500 at about 36ms)
does **not** measure a successful local Jina request in this orb. Port 8081 is
owned by a `websocat` SSH bridge, not llama.cpp. `Searcher.rerankResults`
silently continues after the resulting reranker error while still applying its
fusion/code-path selection, yielding the old numbers.

The Go benchmark was therefore rerun against the explicit, healthy offline
llama.cpp Jina endpoint on port 8084. It independently matched the Python
experiment: 0.667/0.503/0.438 at 3.68s p50. This is both a benchmark-validity
finding and a production observability concern; no production behavior was
changed.

## Cost and deployment

- Full index build: 46m47s wall time on this 8-vCPU CPU-only orb.
- Peak build RSS observed: 3,718,320 KiB (about 3.55 GiB).
- CodeRank weights: 546,938,168 bytes; Hugging Face cache footprint 523 MiB.
- Sidecar vectors: 28,176,512 bytes (about 26.9 MiB), versus the existing
  28 MiB Nomic exact-vector artifact.
- Cached CodeRank query + exact retrieval: 74.7ms p50 / 87.8ms p95, versus
  Nomic's local-server query + exact retrieval at 33.2ms / 57.9ms.
- CodeRank's first-party path is Sentence Transformers/PyTorch. This experiment
  found no first-party llama.cpp or existing Go-native integration; adopting it
  would add a runtime or require an export/runtime validation project.
- Model license: MIT. Local inference is offline after model download.

## Privacy and artifacts

No source or query content was sent to a hosted service. CodeRank, Nomic, and
Jina inference stayed local. Model download is the only network dependency.
Artifacts contain frozen queries, paths, line ranges, rankings, and metrics;
they contain no source bodies or credentials. A post-run secret-pattern and
source-payload scan passed.

- `bench/jev/coderank_benchmark.py`: isolated benchmark scaffold.
- `bench/jev/results-coderankembed.json`: three-run CodeRank/Nomic artifact.
- `bench/jev/results-coderank-jina-control.json`: Go-harness live-Jina control.
- Vector cache (outside repository):
  `~/.cache/sgrep-coderank/coderank-3c4b60807d71-9172.npy`.

## Ranked recommendation

1. **Adopt:** keep Nomic balanced as the local default. It has better aggregate
   quality than CodeRank balanced, better candidate recall, fewer prose errors,
   and lower latency.
2. **Experiment:** first fix reranker success/failure observability, then rerun
   Jina and CodeRank+Jina on a larger blind-human-labeled code-search set. The
   small fixed set suggests CodeRank+Jina can recover recall, but not at an
   interactive CPU latency.
3. **Experiment, hosted opt-in only:** ColBERT + Jev remains the best measured
   aggregate arm (0.750/0.608/0.563), but it uploads queries/source, costs money,
   and has multi-second latency. That is a separate privacy/product decision.
4. **Reject now:** replacing Nomic with CodeRankEmbed or claiming that the old
   0.729 “existing Jina” score proves successful Jina inference.
