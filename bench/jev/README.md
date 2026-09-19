# Isolated Jev reranking experiment

This benchmark-only command routes Jev through the existing
`pkg/rerank.Reranker.Rerank` → `pkg/search.Searcher.rerankResults` boundary. It
does not add a production provider, alter `--rerank`, or change candidate
generation, filtering, deduplication, paths, line ranges, output, or fallback.

## Privacy boundary

Jev is hosted. Running the live arm transmits each query and up to ten
shortlisted source chunks to TypeSafe. The command refuses to run unless both
`TYPESAFE_API_KEY` and the explicit `--allow-source-upload` flag are present.
The output artifact contains paths, ranges, metrics, typed judgments, token
usage, and service errors, but not source text or credentials.

If the API key is unavailable, `--local-only` runs the balanced, local Jina,
and ColBERT arms plus the injected fallback check. It makes no hosted call and
marks Jev quality conclusions as blocked; it is not a substitute for model
evidence.

For isolated self-hosted comparisons, `--local-reranker-endpoint` adds balanced
and ColBERT-shortlist arms through the same existing rerank boundary. The
endpoint must implement llama.cpp's `/v1/rerank` API. Its label is supplied with
`--local-reranker-name`; neither flag changes production configuration.

TypeSafe says requests are not used to train its models. Its public privacy
terms still permit processing/retention needed to provide the service and
telemetry, and zero-data-retention requires separate enterprise terms. Treat
private-source upload as a distinct product/privacy decision.

## Run

Use the pinned `dspy-go` checkout at commit
`87cb50fa0611ef8a3905494c8b16fb0aab7666d3`, index it with the normal local
pipeline, and install the existing local Jina model for the comparison arm.

```bash
sgrep setup --with-rerank
sgrep index /path/to/dspy-go
go run ./bench/jev \
  --corpus /path/to/dspy-go \
  --allow-source-upload \
  --output bench/jev/results.json

# Fully local additional reranker; no source leaves the machine.
go run ./bench/jev \
  --corpus /path/to/dspy-go \
  --local-only \
  --local-reranker-endpoint http://127.0.0.1:8082 \
  --local-reranker-name qwen3-reranker-0.6b-q8 \
  --output bench/jev/results-qwen3-local.json
```

`--jev-score-scale 0.5` is an isolated sensitivity check that shrinks Jev's
score differences halfway toward neutral before the existing fusion boundary.
It does not change the typed rubric or production fusion code. Predeclare a
scale rather than selecting it on held-out results.

The held-out labels are fixed in `dataset.json` before any Jev run and Jev is
not used as its own quality judge. Positive labels are inherited from the
repository's existing model-judged benchmark; they are **not** human-annotated.
The two added no-answer controls and the inherited relevance labels require a
blind human audit before drawing quality conclusions. An injected service
failure verifies exact fail-open equality with the balanced pre-Jev output.
