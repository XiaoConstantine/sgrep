---
title: Storage
description: Understand repository, conversation, model, and server artifacts under the local sgrep home.
weight: 30
---

## Layout

sgrep stores data under `~/.sgrep` unless `SGREP_HOME` selects another root.

```text
~/.sgrep/
├── models/
│   └── nomic-embed-text-v1.5.Q8_0.gguf
├── repos/
│   └── <path-hash>/
│       ├── index.db
│       ├── vectors.mmap
│       ├── vectors.tqmse
│       ├── file_vectors.tqmse
│       ├── colbert_segments.mmap
│       └── metadata.json
├── conversations/
│   ├── conv.db
│   └── turn_embeddings.tqmse
├── server.pid
└── server.log
```

Not every repository has every sidecar. The selected index options, corpus size, and successful completion of each export determine which compatible artifacts exist.

## Repository index

- `index.db` is the source of truth for chunks, file metadata, index compatibility metadata, and FTS/BM25 data.
- `vectors.mmap` holds exact float32 chunk vectors for repositories at or below `--exact-vector-limit`.
- `vectors.tqmse` is the compact TQ-MSE chunk-vector fallback.
- `file_vectors.tqmse` stores compact document-level vectors.
- `colbert_segments.mmap` stores token-segment embeddings for late interaction.
- `metadata.json` maps the hashed directory back to the indexed repository and records index state.

For indexes up to 20,000 chunks, the default index also writes the resident exact-float artifact. Larger indexes use lower-memory TQ-MSE scanning. Set the cutoff with `--exact-vector-limit`; use `0` to disable exact artifacts.

The default compact index does not retain full SQL vector blobs. Add `--sql-vectors` only when you need the legacy SQL search path or the sqlite-vec backend.

## Incremental indexing

`sgrep watch` keeps SQL vectors as incremental-update backing data and refreshes compact sidecars after changes. If a refresh cannot leave compatible sidecars, watch stops with an error rather than serving stale vector artifacts.

All sidecar replacements are built adjacent to the active file and atomically published after validation. Readers retain their previous mapping until a complete replacement is available.

## Conversation index

Conversation metadata, turn content, and FTS data are stored in `conversations/conv.db`. Canonical turn embeddings are stored in the compact `turn_embeddings.tqmse` sidecar. The conversation index records embedding format and context-token metadata transactionally so a configuration change requires a compatible rebuild.

Use `sgrep list`, `sgrep status`, and `sgrep conv status` to inspect stored indexes through the CLI.
