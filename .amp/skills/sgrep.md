# sgrep - Smart Code Search Skill

## Purpose

Use `sgrep` for semantic and hybrid code search when you need to find code by **intent** rather than exact text patterns.

## When to Use This Skill

- Searching for concepts: "error handling", "authentication", "caching logic"
- Searching with specific terms + context: use `--hybrid` for "JWT validation", "OAuth2 token"
- Exploring unfamiliar codebases
- When ripgrep patterns keep missing relevant code
- Finding implementations of features described in natural language

## Setup (One-time)

```bash
# Install llama.cpp
brew install llama.cpp

# Download embedding model
sgrep setup

# Index the codebase
sgrep index .
```

## Commands

| Command | Purpose |
|---------|---------|
| `sgrep "query"` | Semantic search (understands intent) |
| `sgrep --hybrid "query"` | Hybrid search (semantic + BM25 term matching) |
| `sgrep -c "query"` | Search with code context |
| `sgrep --json "query"` | JSON output for parsing |
| `sgrep -q "query"` | Quiet mode (paths only) |
| `sgrep -t "query"` | Include test files |
| `sgrep index .` | Index current directory |
| `sgrep server status` | Check embedding server |

## Semantic vs Hybrid Search

| Mode | Best For | Example Query |
|------|----------|---------------|
| Semantic (default) | Conceptual questions | "how does auth work" |
| Hybrid (`--hybrid`) | Queries with specific terms | "JWT token validation" |

**Use `--hybrid`** when your query contains:
- Function/API names: `--hybrid "parseAST"`
- Technical terms: `--hybrid "OAuth2 refresh token"`
- Specific keywords that should match exactly

**Use semantic (default)** for:
- Conceptual questions: "how is caching implemented"
- Intent-based search: "error handling logic"

## Search Strategy

### The Search Hierarchy

1. **sgrep** → Find relevant files/functions by semantic intent
2. **sgrep --hybrid** → Find code matching intent + specific terms
3. **ast-grep (sg)** → Match structural patterns in those files
4. **ripgrep (rg)** → Exact text for specific symbols

### Example Workflow

```bash
# Step 1: Semantic discovery
sgrep "rate limiting implementation"
# → api/ratelimit.go:20-80

# Step 1b: Or use hybrid for specific terms
sgrep --hybrid "RateLimiter middleware"
# → api/middleware.go:45-90

# Step 2: Structural patterns
sg -p 'rateLimiter.Check($ctx, $key)'

# Step 3: Exact search
rg "RATE_LIMIT_MAX"
```

## Output Interpretation

```bash
$ sgrep "authentication"
auth/middleware.go:45-67      # file:startLine-endLine
auth/jwt.go:12-38
handlers/login.go:89-112
```

Lower scores = more relevant (L2 distance for semantic, hybrid score for `--hybrid`).

## Hybrid Search Tuning

```bash
# Default weights: 60% semantic, 40% BM25
sgrep --hybrid "query"

# More weight on exact term matching
sgrep --hybrid --semantic-weight 0.4 --bm25-weight 0.6 "parseConfig"

# More weight on semantic understanding
sgrep --hybrid --semantic-weight 0.8 --bm25-weight 0.2 "configuration loading"
```

## Tips

- Use natural language queries: "how does the cache invalidation work"
- Use `--hybrid` when searching for specific function names or technical terms
- Combine with `-c` flag to see code snippets
- Use `--json` when parsing results programmatically
- Server auto-starts; use `sgrep server stop` to free resources when done

## Troubleshooting

```bash
# Check server status
sgrep server status

# Re-download model if corrupted
rm -rf ~/.sgrep/models
sgrep setup

# Re-index if results seem stale
sgrep clear
sgrep index .
```
