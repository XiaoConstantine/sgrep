# sgrep - Semantic Code Search Skill

## Purpose

Use `sgrep` for semantic/conceptual code search when you need to find code by **intent** rather than exact text patterns.

## When to Use This Skill

- Searching for concepts: "error handling", "authentication", "caching logic"
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
| `sgrep "query"` | Semantic search |
| `sgrep -c "query"` | Search with code context |
| `sgrep --json "query"` | JSON output for parsing |
| `sgrep -q "query"` | Quiet mode (paths only) |
| `sgrep index .` | Index current directory |
| `sgrep server status` | Check embedding server |

## Search Strategy

### The Search Hierarchy

1. **sgrep** → Find relevant files/functions by semantic intent
2. **ast-grep (sg)** → Match structural patterns in those files
3. **ripgrep (rg)** → Exact text for specific symbols

### Example Workflow

```bash
# Step 1: Semantic discovery
sgrep "rate limiting implementation"
# → api/ratelimit.go:20-80

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

Lower scores = more relevant (L2 distance).

## Tips

- Use natural language queries: "how does the cache invalidation work"
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
