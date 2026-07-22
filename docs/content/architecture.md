---
title: Architecture
description: How sgrep shares local embeddings while keeping code and conversation retrieval pipelines independently tunable.
weight: 30
---

[![sgrep architecture: code and conversation sources feed a shared local embedding service and separate retrieval pipelines](../architecture.jpg)](../architecture.jpg)

[Open the editable tldraw source](../architecture.tldr).

## System shape

Code repositories and coding-agent histories share the same local Nomic embedding model and llama.cpp service. They keep separate stores and retrieval controls because their data shapes and operational needs differ.

The managed-server path owns the local process, verifies its identity and embedding capability, and starts it on demand. Setting `SGREP_ENDPOINT` switches to an externally managed compatible service that sgrep probes but never owns.

## Code indexing

1. Ignore rules remove vendored, generated, and irrelevant files.
2. Go uses the standard-library AST; JavaScript, TypeScript, Python, Rust, C, C++, and Java use tree-sitter-aware chunking. Other text falls back to token-aware splitting.
3. Function, type, symbol, and path metadata enrich both the text sent to embeddings and the lexical index.
4. Every document is formatted with the Nomic `search_document:` task prefix and kept within the shared context budget. Oversized bodies are split before embedding rather than silently truncated.
5. SQLite stores chunk content, file metadata, compatibility metadata, and FTS postings. Exact float32, TQ-MSE, document-vector, and ColBERT sidecars provide retrieval artifacts.

Indexes at or below the default 20,000-chunk cutoff receive an exact float32 mmap for first-stage search. Larger indexes use the compact TQ-MSE artifact. This makes exact scanning the resident speed profile for small and medium repositories while keeping a lower-memory path for large ones.

## Code search

Every query receives the Nomic `search_query:` task prefix. The explicit profile then selects the rest of the pipeline:

```text
fast       query embedding → semantic candidates
balanced   semantic candidates + lexical candidates → weighted RRF
quality    balanced candidates → ColBERT late interaction
optional   selected candidates → generic cross-encoder reranker
```

Lexical retrieval expands code-aware metadata such as camelCase, snake_case, file names, extensions, and paths. Candidate lists from semantic distance and FTS are fused using weighted reciprocal-rank fusion. ColBERT scores token segments with the existing embedding model, so the quality stage requires index storage but no second model.

## Conversation indexing and search

Parsers normalize Claude Code, Codex CLI, Cursor, OpenCode, and Pi histories into canonical sessions and turns. Oversized formatted turns are split within the same context budget, embedded independently, and pooled back to the canonical turn ID.

Conversation metadata and FTS postings remain in their own SQLite database. Compact TQ-MSE turn vectors provide semantic search, while hybrid mode fuses vector and lexical candidates. Results preserve the source session identity so view, export, context extraction, and resume can operate on the original conversation.

## Compatibility and publication

Embedding format, dimensions, context tokens, and artifact metadata are validated when an index opens. Sidecars are written to adjacent temporary files, synchronized, validated, and atomically renamed. A reader keeps the old mapping until a complete replacement can be published, avoiding partial mmap state during rebuilds.
