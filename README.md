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

**Requirements**: llama.cpp (for the embedding server)
```bash
brew install llama.cpp   # macOS
```

## Quick Start

```bash
# One-time setup: downloads embedding model (~130MB)
sgrep setup

# Index your codebase (auto-starts embedding server)
sgrep index .

# Semantic search
sgrep "error handling for database connections"
sgrep "JWT token validation logic"
sgrep "how are API rate limits implemented"

# Watch mode (background indexing)
sgrep watch .
```

The embedding server starts automatically when needed and stays running as a daemon.

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

All data is stored in `~/.sgrep/`:
```
~/.sgrep/
├── models/
│   └── nomic-embed-text-v1.5.Q8_0.gguf   # Embedding model (~130MB)
├── repos/
│   ├── a1b2c3/              # Hash of /path/to/repo1
│   │   ├── index.db         # SQLite + vectors
│   │   └── metadata.json    # Repo path, index time
│   └── d4e5f6/              # Hash of /path/to/repo2
│       └── ...
├── server.pid               # Embedding server PID
└── server.log               # Embedding server logs
```

Use `sgrep list` to see all indexed repositories.

## Commands

| Command | Description |
|---------|-------------|
| `sgrep [query]` | Semantic search (default) |
| `sgrep index [path]` | Index a directory |
| `sgrep watch [path]` | Watch and auto-index |
| `sgrep list` | List all indexed repos |
| `sgrep status` | Show index status |
| `sgrep clear` | Clear index |
| `sgrep setup` | Download model, verify llama-server |
| `sgrep server start` | Manually start embedding server |
| `sgrep server stop` | Stop embedding server |
| `sgrep server status` | Show server status |

## Flags

| Flag | Description |
|------|-------------|
| `-n, --limit N` | Max results (default: 10) |
| `-c, --context` | Show code context |
| `--json` | JSON output for agents |
| `-q, --quiet` | Minimal output (paths only) |
| `--threshold F` | L2 distance threshold (default: 1.5, lower = stricter) |

## Configuration

Environment variables:
```bash
SGREP_HOME=~/.sgrep                    # Data storage location
SGREP_ENDPOINT=http://localhost:8080   # Override embedding server URL
SGREP_PORT=8080                        # Embedding server port
SGREP_DIMS=768                         # Vector dimensions
```

## How It Works

1. **Setup**: `sgrep setup` downloads the embedding model and verifies llama-server
2. **Indexing**: Files are chunked using AST-aware splitting (Go, TS, Python) or size-based fallback
3. **Embedding**: Each chunk is embedded via llama.cpp (local, $0 cost, auto-started)
4. **Storage**: Vectors stored in SQLite, loaded into memory for fast search
5. **Search**: Query embedded → in-memory L2 search → load matching documents

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         sgrep                                │
├──────────────────────────────────────────────────────────────┤
│  Query: "error handling"                                     │
│         ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ llama.cpp   │───▶│ In-Memory   │───▶│   SQLite    │      │
│  │ Embedding   │    │ L2 Search   │    │  Documents  │      │
│  │   (~15ms)   │    │   (~2ms)    │    │   (~5ms)    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│       ▲                                                      │
│       │ Auto-started by sgrep                               │
│       │ (daemon mode, PID tracked)                          │
│                                                              │
│  Total: ~30ms (vs 2800ms with sqlite-vec KNN)               │
└──────────────────────────────────────────────────────────────┘
```

## Performance

Benchmarked on maestro codebase (102 files, 1572 chunks, 768-dim vectors):

| Metric | sgrep | ripgrep | 
|--------|-------|---------|
| Latency (avg) | **31ms** | 10ms |
| Token usage | **57% less** | baseline |
| Attempts needed | 1 | 3-7 |

**Why in-memory search?**

sqlite-vec's KNN queries are slow (~2.9s for 1.5K vectors) due to SQLite's fragmented storage. We load vectors into memory on startup and compute L2 distance in Go, achieving **88x faster** search:

- Vector load: ~95ms (once on startup)
- Embedding: ~15ms (HTTP to llama.cpp)
- L2 search: ~2ms (in-memory)
- Doc fetch: ~5ms (SQLite by ID)

## Chunk Size Limits

The embedding model (nomic-embed-text) has a 2048 token context limit. sgrep handles this by:

1. Default chunk size: 1000 tokens (with AST-aware splitting)
2. Safety truncation at 1500 tokens in embedder
3. Large functions/types split into parts automatically

## License

Apache-2.0
