---
title: Conversation search
description: Search, inspect, export, and resume local histories from Claude Code, Codex, Cursor, OpenCode, and Pi.
weight: 30
---

## Index conversations

Index every discovered source or restrict indexing to one agent:

```bash
sgrep conv index
sgrep conv index --source codex
sgrep conv index --source claude
sgrep conv index --source cursor
sgrep conv index --source opencode
sgrep conv index --source pi
```

Use watch mode to ingest new sessions, or force a full refresh of cached sessions:

```bash
sgrep conv index --watch
sgrep conv index --force
```

## Search and filter

The shorthand and explicit search forms are equivalent:

```bash
sgrep conv "authentication cache key"
sgrep conv search "authentication cache key"
```

Narrow results by source, project, or time:

```bash
sgrep conv "session compaction" --agent codex --limit 5
sgrep conv "database migration" --agent claude --since 7d
sgrep conv "bug fix" --project payment-service --after 2026-01-01 --before 2026-06-01
```

Semantic search is the default. Add `--hybrid` to combine it with conversation FTS, or `--exact` for lexical-only retrieval:

```bash
sgrep conv "JWT token" --hybrid
sgrep conv "JWT_SECRET" --exact
```

## Inspect and reuse a session

Search output includes the canonical session ID. Use it to view, export, resume, or extract context:

```bash
sgrep conv view <session_id>
sgrep conv view <session_id> --turn 3 --no-color

sgrep conv export <session_id> --format markdown -o conversation.md
sgrep conv export <session_id> --format json -o conversation.json

sgrep conv resume <session_id>
sgrep conv context <session_id> --turns 10 --copy
```

Copy a whole conversation, a single turn, or only its code blocks:

```bash
sgrep conv copy <session_id>
sgrep conv copy <session_id> --turn 2 --code-only
```

Use `sgrep conv status` to inspect the conversation index.

## Storage and privacy

Conversation content and metadata are stored locally in `~/.sgrep/conversations/conv.db`; compact turn vectors live in `~/.sgrep/conversations/turn_embeddings.tqmse`. Re-running `sgrep conv index` backfills missing embeddings and refreshes the vector sidecar.
