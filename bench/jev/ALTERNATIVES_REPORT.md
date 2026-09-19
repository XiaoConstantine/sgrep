# Self-hosted reranker alternatives (exploratory)

## Conclusion

Do not replace the current Nomic balanced default or adopt a local reranker
from this evidence. The strongest code-specialized candidate found,
`lightonai/LateOn-Code`, was also tested: it tied verified live Jina's quality
through the existing shortlist boundary, took 13.0s p50 on CPU, and its direct
ranking was materially worse. The two strongest llama.cpp-compatible
cross-encoders likewise did not beat verified Jina or ColBERT.

All quality numbers are exploratory. The ten-query held-out set retains its
fixed labels, but those labels have not received the requested blind human
audit. No candidate model or Jev was used as a judge.

## Local benchmark

Environment: 8 vCPU, 15 GiB RAM, no GPU; llama.cpp `0.4.0-dev`, build 10809,
commit `5266f24da`. Each arm preserves the existing candidates, filters,
deduplication, paths/ranges, output and fusion boundary. The two added models
ran via `/v1/rerank` on `127.0.0.1`, with `--offline`, 8 threads, 8192 context,
and 2048 logical/physical batch size. Each model arm ran three times.

| Arm | MRR | NDCG@10 | R@10 | Prose promotion errors | p50 / p95 end-to-end |
|---|---:|---:|---:|---:|---:|
| Balanced | 0.667 | 0.499 | 0.438 | 1 | 35.4 / 48.8 ms |
| Balanced + verified live Jina | 0.667 | 0.503 | 0.438 | 1 | 3,641 / 5,504 ms |
| Quality / ColBERT | **0.729** | **0.567** | **0.500** | 1 | 6,645 / 8,638 ms |
| Balanced + LateOn-Code rank fusion | 0.667 | 0.503 | 0.438 | 1 | 12,991 / 18,980 ms |
| LateOn-Code direct top-20 order (diagnostic) | 0.573 | 0.448 | 0.406 | 4 | same inference |
| Balanced + BGE v2-m3 Q4_K_M | 0.667 | 0.503 | 0.438 | 1 | 10,874 / 14,881 ms |
| ColBERT + BGE v2-m3 Q4_K_M | 0.688 | 0.518 | 0.438 | **0** | 17,590 / 23,488 ms |
| Balanced + Qwen3 0.6B Q8 | 0.667 | 0.503 | 0.438 | 1 | 19,079 / 26,954 ms |
| ColBERT + Qwen3 0.6B Q8 | 0.713 | 0.562 | 0.500 | **0** | 25,039 / 36,004 ms |
| Balanced + Jev (prior live run) | 0.708 | 0.553 | 0.500 | 2 | 452 / 521 ms |
| ColBERT + Jev (prior live run) | **0.750** | **0.608** | **0.563** | 1 | 7,179 / 9,102 ms |

The local models produced identical metrics on all three corrected runs.
Qwen's GGUF was 639,153,184 bytes, peaked at 2,886,880 KiB RSS, and processed
367 prompt tokens/s. BGE's Q4_K_M was 438,376,864 bytes, peaked at 1,794,828
KiB RSS, and processed 667 prompt tokens/s. These are single-server CPU
measurements, not cross-hardware throughput claims.

The generic answerability thresholds were fixed before inspecting scores:
Qwen probability >= 0.5 classified 10/10 cases correctly; BGE raw logit >= 0
classified 9/10, rejecting both no-answer controls but also rejecting the
positive self-refine case. This diagnostic scored only each arm's returned top
10 and does not establish calibrated corpus-level rejection.

On six independently authored synthetic implementation/prose pairs, Qwen chose
implementation in 2/6 and BGE in 1/6. A single predeclared code-oriented query
instruction left Qwen at 2/6 and was not evaluated against held-out labels.

LateOn-Code was pinned at revision
`ace431824f35db231178fc602e33296784762a2e` and run locally with PyLate 1.3.4.
It is a 149M ModernBERT late-interaction model trained on CoRNStack, CoIR, and
CodeSearchNet including Go. Its Apache-2.0 model card reports MTEB Code 74.12
and CodeSearchNet MaxSim NDCG@10 0.8942; those are author-reported numbers.
On this frozen transfer set, three production-compatible rank-fusion runs were
deterministic, had no inference failures/fallbacks, peaked at 2,321,672 KiB RSS,
and tied live Jina exactly. A separate direct-order diagnostic was worse on all
three quality metrics and increased prose-promotion errors from one to four.
It returned results for both no-answer controls and has no calibrated
answerability threshold. The model cache was 573 MiB; the isolated Python
environment was 7.0 GiB because PyTorch installed CUDA runtime packages even
though inference used CPU.

## Runtime validity and failure behavior

Initial runs with llama.cpp's default 512-token physical batch were invalid:
longer chunks returned HTTP 500 while `Searcher.rerankResults` intentionally
continued without that reranker signal. Those files are retained with
`invalid-ubatch512` in their names and excluded. Corrected servers used
`--batch-size 2048 --ubatch-size 2048`; post-run full-top-10 protocol checks
returned every score for 10/10 queries for each model.

The benchmark still verifies the scaffold's injected Jev failure gives exact
pre-Jev order. The production search boundary swallows a local reranker error
and continues, but can allocate reranker fusion weight with no scores; that is
not always exact pre-reranker order. No production code was changed here.

## Focused survey

| Candidate | Evidence and fit | Deployment / license | Recommendation |
|---|---|---|---|
| LateOn-Code | Best code-specific hypothesis: 149M multi-vector model trained on CoRNStack/CoIR/CodeSearchNet, author-reported MTEB Code 74.12. Local fused quality only tied verified Jina; direct ranking regressed to 0.573/0.448/0.406 with four prose errors. | 2,048 document / 256 query tokens, 128-dimensional token vectors; PyLate, ONNX and FastPLAID/ColGrep; Apache-2.0. Local CPU p50 13.0s and 2.21 GiB peak RSS. | **Most promising model found, but reject adoption on measured transfer.** Re-test only on a larger blind-human-labeled set or GPU. |
| Qwen3-Reranker-0.6B | Official card claims MTEB-Code 73.42 versus Jina 58.98 and BGE 41.38, but these are Qwen-authored runs over Qwen dense candidates. Local fixed-label quality did not beat Jina; implementation diagnostic 2/6. | 0.6B, 32K, 100+ natural/programming languages; Transformers, vLLM, llama.cpp GGUF; Apache-2.0. | **Experiment on a larger human-labeled set only.** Strongest surveyed llama.cpp cross-encoder candidate, but reject current CPU adoption. |
| BGE-reranker-v2-m3 | Mature multilingual XLM-R cross-encoder. Jina's author-run CodeSearchNet table reports BGE 62.86 versus Jina 71.36. Local quality regressed and diagnostic was 1/6. | 568M; Transformers, FlagEmbedding, TEI, llama.cpp/ONNX ecosystem; Apache-2.0. | **Reject for sgrep's current shortlist.** |
| Existing Jina v2 | Best local quality/latency tradeoff in this experiment and author-reported CodeSearchNet 71.36. Its one prose error remains. | 278M, 1024 tokens plus sliding window; Transformers/ONNX/GGUF; **CC-BY-NC-4.0**, commercial use needs separate terms. | **Keep only where the noncommercial license is acceptable.** License is the main adoption blocker. |
| CodeRankEmbed | Code-specific 137M bi-encoder; author reports CSN MRR 77.9 and CoIR NDCG@10 60.1. Independent 2026 CORE-Bench finds it stronger than BGE-M3 among small off-the-shelf embeddings, but far below in-domain models. It changes first-stage embeddings rather than acting as a drop-in cross-encoder. | 8,192 context, required query prefix; Sentence Transformers; no first-party llama.cpp rerank route; MIT. Requires re-indexing and separate Go/runtime work. | **Highest-priority retrieval experiment**, not a reranker replacement. |
| CodeRankLLM | Code-specific listwise model, but training order was supervised by Qwen2.5-32B and the card supplies no compact runtime benchmark. | 7B/8B-class BF16, community GGUFs, likely several GiB and slow on this CPU; MIT. | **Reject locally; GPU-only research candidate.** |
| answerai-colbert-small-v1 | Very compact 33.4M, 96-dimensional token vectors, author-reported BEIR 53.79. No persuasive code-specific evidence. | 512 positions; Sentence Transformers MultiVector, ONNX, RAGatouille/ColBERT; Apache-2.0. Requires a new late-interaction index/runtime rather than the current llama endpoint. | **Low-cost experiment only** if replacing the current ColBERT implementation is independently desired. |
| mxbai-rerank-base-v2 | Claims code support and 31.73 code-search score, but its card's latency is on A100 and its evidence is model-author supplied. | 0.5B, 109 languages; Transformers/Sentence Transformers/TEI; Apache-2.0. No validated drop-in llama.cpp path in this experiment. | **Defer behind Qwen/CodeRankEmbed.** |
| Compact coder LLM as classifier | llama.cpp-compatible and rubric-capable, but lacks reranker calibration and adds generation/parsing latency. Qwen3 already took 19 s p50 as a purpose-trained 0.6B reranker here. | Many Apache-2.0 Qwen coder sizes; llama.cpp/ONNX varies. | **Reject** for interactive CPU search. |

No credible independent benchmark located compares these rerankers directly on
implementation-versus-prose code search. Model-card tables are therefore marked
as author/vendor evidence. CORE-Bench is independent and code-focused but
benchmarks retrievers/embeddings, not this cross-encoder reranking boundary.

## Privacy and integration

The alternatives are fully offline after model download. Query/source chunks
remain on-device, and the server bound only to loopback with llama.cpp offline
mode. Model files live outside the repository under
`~/.cache/sgrep-rerank-research/`. Artifacts contain queries, paths, line ranges,
scores and metrics, but no source contents or credentials. A privacy scan found
no API-key-shaped values or environment secret values.

Go integration for BGE/Qwen is mechanically simple through the existing
`pkg/rerank.Reranker`, but model-specific score semantics matter: BGE emits raw
logits while Qwen's llama.cpp conversion emits probabilities, and current
fusion applies sigmoid uniformly. A production experiment would first need an
explicit score contract and observable per-query reranker failures. Neither was
added in this isolated benchmark.

## Sources

- Qwen3 model card and author benchmark: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Official ggml Qwen Q8 conversion: <https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF>
- BGE model card: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- BGE GGUF conversion: <https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF>
- Jina v2 model card: <https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual>
- CodeRankEmbed and CodeRankLLM: <https://huggingface.co/nomic-ai/CodeRankEmbed>, <https://huggingface.co/nomic-ai/CodeRankLLM>
- AnswerAI ColBERT: <https://huggingface.co/answerdotai/answerai-colbert-small-v1>
- Mixedbread v2: <https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2>
- llama.cpp reranking endpoint/flags: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Independent CORE-Bench (retrieval, not reranking): <https://arxiv.org/abs/2606.11864>
- LateOn-Code model card and benchmark: <https://huggingface.co/lightonai/LateOn-Code>
- PyLate runtime: <https://github.com/lightonai/pylate>
