# sgrep - Smart Code Search

Use `sgrep` for semantic and hybrid code search. It understands intent, not just exact strings.

## When to Use

- Finding code by **concept**: "error handling", "authentication logic", "rate limiting"
- Searching for **specific terms** with semantic context: use `--hybrid`
- Exploring unfamiliar codebases
- When ripgrep patterns keep missing relevant code

## Quick Reference

```bash
# First time only
sgrep setup

# Index current directory
sgrep index .

# Semantic search (understands intent)
sgrep "database connection pooling"
sgrep "how are errors handled"

# Hybrid search (semantic + exact term matching via BM25)
sgrep --hybrid "JWT validation"
sgrep --hybrid "authentication middleware"

# Tune hybrid weights (default: 60% semantic, 40% BM25)
sgrep --hybrid --semantic-weight 0.5 --bm25-weight 0.5 "error handler"

# With code context
sgrep -c "authentication middleware"

# JSON output (for parsing)
sgrep --json "error handling"

# Quiet mode (paths only)
sgrep -q "logging"

# Include test files
sgrep -t "mock database"
```

## Semantic vs Hybrid

| Mode | Best For | Example |
|------|----------|---------|
| Semantic (default) | Conceptual queries | "how does auth work" |
| Hybrid (`--hybrid`) | Queries with specific terms | "JWT token validation" |

**Use hybrid when** your query contains exact technical terms (function names, APIs, specific keywords) that should be matched literally alongside semantic understanding.

## Search Hierarchy

1. **sgrep** → Find files/functions by intent (semantic) or intent + terms (hybrid)
2. **ast-grep** → Match structural patterns
3. **ripgrep** → Exact text search

## Example Workflow

```bash
# Find authentication code semantically
sgrep "user authentication flow"
# → auth/handler.go:45-80

# Or use hybrid for specific term matching
sgrep --hybrid "OAuth2 token refresh"
# → auth/oauth.go:120-150

# Then use ast-grep for structural patterns
sg -p 'if err != nil { return $_ }'

# Then ripgrep for specific symbols
rg "JWT_SECRET"
```

## Flags

| Flag | Description |
|------|-------------|
| `-n, --limit N` | Max results (default: 10) |
| `-c, --context` | Show code context |
| `--json` | JSON output |
| `-q, --quiet` | Paths only |
| `-t, --include-tests` | Include test files |
| `--hybrid` | Enable hybrid search (semantic + BM25) |
| `--semantic-weight` | Weight for semantic (default: 0.6) |
| `--bm25-weight` | Weight for BM25 (default: 0.4) |
| `--threshold` | Distance threshold (default: 1.5) |

## Server Management

The embedding server auto-starts. Manual control:

```bash
sgrep server status   # Check status
sgrep server stop     # Stop server
```
