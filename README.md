# sgrep - Semantic Grep for Code

**Semantic code search that complements `ripgrep` and `ast-grep`.**

```
┌─────────────────────────────────────────────────────────────────┐
│  ripgrep (rg)     │  ast-grep (sg)    │  sgrep              │
│  ─────────────    │  ──────────────   │  ──────             │
│  Exact text/regex │  AST patterns     │  Semantic intent    │
│  "findUser"       │  $fn($args)       │  "auth validation"  │
└─────────────────────────────────────────────────────────────────┘
```

## Why sgrep?

Coding agents (Amp, Claude Code, Cursor) waste tokens on failed `grep` attempts when searching for concepts rather than exact strings. `sgrep` understands **what you mean**, not just what you type.

```bash
# ❌ Agent tries 10+ grep patterns, burns 2000 tokens
rg "authenticate" && rg "auth" && rg "login" && rg "session" ...

# ✅ One semantic query, 50 tokens
sgrep "how does user authentication work"
```

## Installation

```bash
go install github.com/XiaoConstantine/sgrep@latest
```

**Requirements**: llama.cpp server running with an embedding model:
```bash
# Start llama.cpp with nomic-embed-text (768 dims)
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --port 8080
```

## Quick Start

```bash
# Index your codebase (creates .sgrep/index.db)
sgrep index .

# Semantic search
sgrep "error handling for database connections"
sgrep "JWT token validation logic"
sgrep "how are API rate limits implemented"

# Watch mode (background indexing)
sgrep watch .
```

## Agent-Optimized Output

Default output is minimal for token efficiency:

```bash
$ sgrep "authentication middleware"
auth/middleware.go:45-67
auth/jwt.go:12-38
handlers/login.go:89-112
```

Use `-c` for context (still concise):
```bash
$ sgrep -c "authentication middleware"
auth/middleware.go:45-67
  func AuthMiddleware(next http.Handler) http.Handler {
      token := r.Header.Get("Authorization")
      ...

auth/jwt.go:12-38
  func ValidateJWT(token string) (*Claims, error) {
      ...
```

JSON output for programmatic use:
```bash
$ sgrep --json "authentication"
[{"file":"auth/middleware.go","start":45,"end":67,"score":0.92}]
```

## Combining with ripgrep and ast-grep

**The search hierarchy for agents:**

1. **sgrep** - Find the right files/functions by intent
2. **ast-grep** - Match structural patterns in those files  
3. **ripgrep** - Exact text search for specific symbols

Example workflow:
```bash
# Step 1: Semantic search to find relevant code
sgrep "rate limiting implementation" 
# → api/ratelimit.go:20-80

# Step 2: AST pattern to find all similar usages
sg -p 'rateLimiter.Check($ctx, $key)' 

# Step 3: Exact search for specific constant
rg "RATE_LIMIT_MAX"
```

## Storage

All indexes are stored in `~/.sgrep/`:
```
~/.sgrep/
├── repos/
│   ├── a1b2c3/              # Hash of /path/to/repo1
│   │   ├── index.db         # SQLite + vectors
│   │   └── metadata.json    # Repo path, index time
│   └── d4e5f6/              # Hash of /path/to/repo2
│       └── ...
└── cache/                   # Embedding cache (future)
```

Use `sgrep list` to see all indexed repositories.

## Configuration

Environment variables:
```bash
SGREP_HOME=~/.sgrep                    # Index storage location
SGREP_ENDPOINT=http://localhost:8080   # llama.cpp server
SGREP_MODEL=nomic-embed-text           # embedding model
SGREP_DIMS=768                         # vector dimensions
```

## Commands

| Command | Description |
|---------|-------------|
| `sgrep [query]` | Semantic search (default) |
| `sgrep index [path]` | Index a directory |
| `sgrep watch [path]` | Watch and auto-index |
| `sgrep list` | List all indexed repos |
| `sgrep status` | Show index status |
| `sgrep clear` | Clear index |

## Flags

| Flag | Description |
|------|-------------|
| `-n, --limit N` | Max results (default: 10) |
| `-c, --context` | Show code context |
| `--json` | JSON output for agents |
| `-q, --quiet` | Minimal output (paths only) |
| `--threshold F` | Similarity threshold (0-1) |

## How It Works

1. **Indexing**: Files are chunked using AST-aware splitting (Go, TS, Python) or size-based fallback
2. **Embedding**: Each chunk is embedded via llama.cpp (local, $0 cost)
3. **Storage**: Vectors stored in SQLite with sqlite-vec (int8 quantized)
4. **Search**: Query embedded → vector similarity search → ranked results

## Performance

- **Indexing**: ~100 files/sec with llama.cpp on M1
- **Search**: <50ms for 10K chunks
- **Storage**: ~1KB per chunk (quantized vectors)
- **Memory**: ~50MB for 10K file index

## License

Apache-2.0
