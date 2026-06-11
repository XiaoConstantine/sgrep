---
name: sgrep
description: Semantic and hybrid code and conversation search for intent-based queries. Use when exploring unfamiliar codebases, finding code by concept instead of exact text, or recalling past agent conversations about similar problems.
license: Apache-2.0
compatibility: Requires the sgrep binary; semantic and conversation indexing use a local llama.cpp-compatible embedding server.
metadata:
  homepage: https://github.com/XiaoConstantine/sgrep
---

# sgrep - Smart Code & Conversation Search

Use `sgrep` for semantic and hybrid search across **code** and **agent conversations**. It understands intent, not just exact strings.

## When to Use

### Code Search
- Finding code by **concept**: "error handling", "authentication logic", "rate limiting"
- Searching for **specific terms** with semantic context: use `--hybrid`
- Best code-search accuracy after indexing: use `--hybrid --colbert`
- Exploring unfamiliar codebases
- When ripgrep patterns keep missing relevant code

### Conversation Search
- Finding past discussions with **Claude Code**, **Codex CLI**, **Cursor**, **OpenCode**, or **Pi**
- Recalling how you solved a similar problem before
- Building context from previous sessions for new tasks
- Searching across all your coding agent interactions

## Commands

```bash
# First time only
sgrep setup
sgrep setup --with-rerank  # optional, only for --rerank

# Index current directory; builds compact TQ-MSE chunk/file vectors by default
sgrep index .

# Optional ColBERT segment codec override
sgrep index . --colbert-codec tqmse
sgrep index . --colbert-codec int8
sgrep index . --colbert-codec pq6

# Legacy compatibility: also persist full SQL vectors
sgrep index . --sql-vectors

# Watch mode keeps SQL vectors for incremental updates; rerun index to compact
sgrep watch .

# Semantic code search
sgrep "database connection pooling"
sgrep "how are errors handled"

# Hybrid search for specific terms + context
sgrep --hybrid "JWT validation"
sgrep --hybrid "authentication middleware"

# Best code-search accuracy
sgrep --hybrid --colbert "JWT validation"
sgrep --hybrid --colbert "authentication middleware"

# With code context
sgrep -c "authentication middleware"

# JSON output
sgrep --json "rate limiting"
```

## Conversation Search

```bash
# Index conversations; refreshes compact TQ-MSE turn vectors
sgrep conv index
sgrep conv index --source codex
sgrep conv index --source claude
sgrep conv index --source opencode
sgrep conv index --source pi
sgrep conv index --watch
sgrep conv index --force

# Search conversations
sgrep conv "authentication flow"
sgrep conv "JWT refresh_token" --hybrid
sgrep conv "database migration" --agent claude --since 7d
sgrep conv "bug fix" --project payment-service --after 2026-01-01 --before 2026-06-01
sgrep conv "exact phrase" --exact
sgrep conv "auth" --json -n 1

# View, export, context, and copy helpers
sgrep conv view <session_id>
sgrep conv view <session_id> --turn 3 --no-color
sgrep conv export <session_id> --format markdown -o conversation.md
sgrep conv export <session_id> --format json -o conversation.json
sgrep conv context <session_id>
sgrep conv context <session_id> --turns 10 --copy
sgrep conv copy <session_id> --turn 2 --code-only
sgrep conv status
```

## Semantic vs Hybrid

| Mode | Best For | Example |
|------|----------|---------|
| Semantic (default) | Conceptual queries | "how does auth work" |
| `--hybrid` | Queries with specific terms | "JWT token validation" |
| `--hybrid --colbert` | Highest code-search accuracy | "authentication middleware" |

**Use `--hybrid`** when your query contains function names, API names, or technical terms that should match exactly.
Use `--hybrid --colbert` when accuracy matters more than the fastest possible query.

## Search Hierarchy

1. **sgrep** → Find relevant files/functions by semantic intent
2. **sgrep --hybrid** → Find code matching intent + specific terms
3. **sgrep --hybrid --colbert** → Rerank candidates with late interaction for best code accuracy
4. **ast-grep** → Match structural patterns in those files
5. **ripgrep** → Exact text for specific symbols
