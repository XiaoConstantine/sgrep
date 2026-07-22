---
title: Code search
description: Index repositories and choose semantic, lexical, late-interaction, and reranking controls deliberately.
weight: 20
---

## Index a repository

Run indexing from a repository or pass its path explicitly:

```bash
sgrep index .
sgrep index /path/to/repository
```

Indexing performs AST-aware chunking where supported, embeds chunks and files locally, stores source metadata and FTS data in SQLite, and writes compact vector sidecars. ColBERT segments are precomputed by default for the `quality` profile.

Use watch mode while a repository is changing:

```bash
sgrep watch .
```

Watch mode incrementally updates SQL vectors and refreshes compact sidecars after changes. Run a full `sgrep index` periodically to rebuild a fully compact index.

## Retrieval profiles

Profiles select reproducible pipelines rather than changing behavior based on whichever artifacts happen to exist.

| Profile | Pipeline | Best for |
|---------|----------|----------|
| `fast` | Semantic | Low-latency exploration |
| `balanced` | Semantic + BM25 | Everyday search; the default |
| `quality` | Semantic + BM25 + ColBERT | Harder ranking problems |

```bash
sgrep --profile fast "request routing"
sgrep --profile balanced "authentication middleware"
sgrep --profile quality "JWT token validation logic"
```

Hybrid retrieval fuses semantic and lexical candidates with weighted reciprocal-rank fusion. You can override the profile controls when experimenting:

```bash
sgrep --hybrid --semantic-weight 0.4 --bm25-weight 0.6 "parseAST"
sgrep --profile balanced --colbert "authentication middleware"
sgrep --profile quality --no-colbert "authentication middleware"
```

ColBERT uses the embedding model already installed by `sgrep setup`; it does not require a second model.

## Cross-encoder reranking

Generic cross-encoder reranking is experimental and off by default for code search. Install its separate model only if you intend to test it:

```bash
sgrep setup --with-rerank
sgrep --profile quality --rerank "authentication middleware"
```

## Output for people and agents

Default output is intentionally compact:

```text
auth/middleware.go:45-67
auth/jwt.go:12-38
```

Choose context, paths-only output, or JSON explicitly:

```bash
sgrep -c "authentication middleware"
sgrep -q "authentication middleware"
sgrep --json "authentication middleware"
```

Tests are excluded by default. Add `--include-tests` when test code is part of the search target, and use `--all-chunks` to disable per-file result deduplication.

## Repository-level questions

sgrep also stores a document embedding for each file. Meta-queries such as these can therefore rank overview documents rather than only implementation chunks:

```bash
sgrep "what does this repository do"
sgrep "project overview"
```
