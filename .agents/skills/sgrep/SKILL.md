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
- Exploring unfamiliar codebases
- When ripgrep patterns keep missing relevant code

### Conversation Search
- Finding past discussions with **Claude Code**, **Codex CLI**, or **Cursor**
- Recalling how you solved a similar problem before
- Building context from previous sessions for new tasks
- Searching across all your coding agent interactions

## Commands

```bash
# First time only
sgrep setup

# Index current directory
sgrep index .

# Semantic code search
sgrep "database connection pooling"
sgrep "how are errors handled"

# Hybrid search for specific terms + context
sgrep --hybrid "JWT validation"
sgrep --hybrid "authentication middleware"

# With code context
sgrep -c "authentication middleware"

# JSON output
sgrep --json "rate limiting"
```

## Conversation Search

```bash
# Index conversations
sgrep conv index

# Search conversations
sgrep conv "authentication flow"
sgrep conv "JWT refresh_token" --hybrid
sgrep conv "database migration" --agent claude --since 7d

# View or export a session
sgrep conv view <session_id>
sgrep conv context <session_id>
```

## Semantic vs Hybrid

| Mode | Best For | Example |
|------|----------|---------|
| Semantic (default) | Conceptual queries | "how does auth work" |
| `--hybrid` | Queries with specific terms | "JWT token validation" |

**Use `--hybrid`** when your query contains function names, API names, or technical terms that should match exactly.

## Search Hierarchy

1. **sgrep** → Find relevant files/functions by semantic intent
2. **sgrep --hybrid** → Find code matching intent + specific terms
3. **ast-grep** → Match structural patterns in those files
4. **ripgrep** → Exact text for specific symbols
