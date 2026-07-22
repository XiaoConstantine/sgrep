---
title: CLI reference
description: Code-search, indexing, conversation, setup, server, and integration commands.
weight: 10
---

## Commands

| Command | Purpose |
|---------|---------|
| `sgrep [query]` | Search the current repository with the balanced profile |
| `sgrep index [path]` | Build or rebuild a repository index |
| `sgrep watch [path]` | Watch a repository and update its index |
| `sgrep list` | List indexed repositories |
| `sgrep status` | Show repository index status |
| `sgrep clear` | Clear a repository index |
| `sgrep conv [search] <query>` | Search indexed agent conversations |
| `sgrep conv index` | Index or watch conversation sources |
| `sgrep conv view <session_id>` | View a conversation |
| `sgrep conv export <session_id>` | Export a conversation |
| `sgrep conv context <session_id>` | Extract reusable context |
| `sgrep conv copy <session_id>` | Copy a conversation or turn |
| `sgrep conv resume <session_id>` | Resume in the originating agent when supported |
| `sgrep conv status` | Show conversation index status |
| `sgrep setup` | Download the embedding model and verify llama.cpp |
| `sgrep server start\|stop\|status` | Manage the local embedding daemon explicitly |
| `sgrep install-claude-code` | Install Claude Code hooks and skill |
| `sgrep install-skill` | Install offline skill copies |

Run `sgrep <command> --help` for the authoritative flags accepted by the installed version.

## Code-search flags

| Flag | Meaning |
|------|---------|
| `-n, --limit N` | Maximum results; default 10 |
| `-c, --context` | Include code context |
| `--json` | Emit machine-readable JSON |
| `-q, --quiet` | Emit paths only |
| `--threshold F` | Maximum cosine distance; lower is stricter |
| `-t, --include-tests` | Include test files |
| `--all-chunks` | Disable per-file result deduplication |
| `--profile fast\|balanced\|quality` | Select the retrieval pipeline |
| `--hybrid` | Enable semantic and BM25 retrieval |
| `--semantic-weight F` | Semantic weight in hybrid fusion; default 0.6 |
| `--bm25-weight F` | BM25 weight in hybrid fusion; default 0.4 |
| `--colbert` | Enable ColBERT late interaction |
| `--no-colbert` | Disable ColBERT even when the profile selects it |
| `--colbert-adaptive-segments` | Experiment with adaptive segment budgets |
| `--rerank` | Enable cross-encoder reranking |
| `--rerank-topk N` | Candidate count passed to reranking; default 50 |
| `-d, --debug` | Print timing diagnostics; repeat for more detail |
| `--debug-log PATH` | Write debug events to a file |
| `--trace` | Emit a search trace |

## Index flags

| Flag | Meaning |
|------|---------|
| `--workers N` | Parallel embedding workers; 0 selects the default |
| `--quantize none\|int8\|binary` | Legacy SQL-vector quantization mode |
| `--sql-vectors` | Retain full first-stage vectors in SQL for legacy search |
| `--colbert-preindex` | Precompute ColBERT segments; enabled by default |
| `--colbert-adaptive-segments` | Use adaptive segment budgets during preindexing |
| `--colbert-codec tqmse\|int8\|pq6` | Select the ColBERT sidecar codec |
| `--exact-vector-limit N` | Exact-float artifact cutoff; default 20,000, 0 disables |

## Conversation-search flags

| Flag | Meaning |
|------|---------|
| `-n, --limit N` | Maximum results; default 10 |
| `-T, --threshold F` | Similarity threshold from 0 to 1 |
| `--hybrid` | Combine semantic and keyword retrieval |
| `--exact` | Use keyword matching only |
| `-a, --agent NAME` | Filter by Claude, Codex, Cursor, OpenCode, Pi, or all |
| `-p, --project VALUE` | Filter by project name or path |
| `--since DURATION` | Relative time filter such as `7d` or `2w` |
| `--after DATE` | Include conversations on or after `YYYY-MM-DD` |
| `--before DATE` | Include conversations on or before `YYYY-MM-DD` |
| `--json` | Emit JSON |
| `-q, --quiet` | Emit session IDs only |
| `-v, --verbose` | Include full turn content |
| `--format default\|table\|timeline` | Select presentation format |
| `-i, --interactive` | Enter interactive mode after search |

Indexing adds `--source`, `--force`, and `--watch`. View, export, context, copy, and resume each expose focused flags through their own `--help` output.
