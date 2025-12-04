# sgrep - Smart Grep for Code

**Semantic + hybrid code search that complements `ripgrep` and `ast-grep`.**

```
┌─────────────────────────────────────────────────────────────────┐
│  ripgrep (rg)     │  ast-grep (sg)    │  sgrep              │
│  ─────────────    │  ──────────────   │  ──────             │
│  Exact text/regex │  AST patterns     │  Semantic + hybrid  │
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

### Homebrew (macOS/Linux)

```bash
brew tap XiaoConstantine/tap
brew install sgrep
```

### Quick Install (curl)

```bash
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/sgrep/main/install.sh | bash
```

### Go Install

```bash
go install github.com/XiaoConstantine/sgrep/cmd/sgrep@latest
```

### From Source

```bash
git clone https://github.com/XiaoConstantine/sgrep.git
cd sgrep

# Default build (uses libSQL with DiskANN vector search)
go build -o sgrep ./cmd/sgrep

# Alternative: sqlite-vec backend
go build -tags=sqlite_vec -o sgrep ./cmd/sgrep
```

**Requirements**: llama.cpp (for the embedding server)
```bash
brew install llama.cpp   # macOS
# or build from source: https://github.com/ggerganov/llama.cpp
```

### As Library

```bash
go get github.com/XiaoConstantine/sgrep@latest
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

# Hybrid search (semantic + BM25) - better for exact term matching
sgrep --hybrid "authentication middleware"
sgrep --hybrid "JWT token" --semantic-weight 0.5 --bm25-weight 0.5

# Watch mode (background indexing)
sgrep watch .
```

The embedding server starts automatically when needed and stays running as a daemon.

## Hybrid Search

Hybrid search combines **semantic understanding** with **lexical matching (BM25)** for improved accuracy. This helps when:
- Searching for specific technical terms (e.g., "JWT", "OAuth", "mutex")
- The query contains exact function/variable names
- Semantic search alone misses exact keyword matches

```bash
# Default: semantic-only search
sgrep "authentication"

# Hybrid: semantic (60%) + BM25 (40%) - default weights
sgrep --hybrid "authentication"

# Custom weights: more emphasis on exact matches
sgrep --hybrid --semantic-weight 0.4 --bm25-weight 0.6 "parseAST"
```

**Note**: Hybrid search requires building with FTS5 support (see [From Source](#from-source)). The FTS5 index is created automatically on first hybrid search - no re-indexing needed.

## Two-Stage Retrieval with Reranking

For higher precision, sgrep supports **cross-encoder reranking** using a dedicated reranker model:

```bash
# Enable reranking for better precision
sgrep --rerank "error handling"

# Combine with hybrid search
sgrep --rerank --hybrid "authentication middleware"
```

**How it works:**
1. **Stage 1**: Fast vector search retrieves top candidates (~30 docs)
2. **Stage 2**: Cross-encoder reranks candidates using query-document attention

**Setup reranking:**
```bash
# Download reranker model (~636MB) and start server
sgrep setup --with-rerank
```

The reranker uses BGE-reranker-v2-m3 via llama.cpp, running on a separate port (8081). It provides better precision for complex queries at the cost of ~300-700ms additional latency.

## Document-Level Search

sgrep automatically handles meta-queries about your repository:

```bash
# These queries use document-level embeddings
sgrep "what does this repo do"
sgrep "project overview"
sgrep "purpose of this codebase"
```

Document-level embeddings (mean of chunk embeddings per file) are computed during indexing, enabling README.md and other overview files to rank highly for repository-level questions.

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
│   │   ├── index.db         # libSQL database with DiskANN vectors
│   │   └── metadata.json    # Repo path, index time
│   └── d4e5f6/              # Hash of /path/to/repo2
│       └── ...
├── server.pid               # Embedding server PID
└── server.log               # Embedding server logs
```

Use `sgrep list` to see all indexed repositories.

## Storage Backends

sgrep supports two vector storage backends:

| Backend | Build Command | Storage Efficiency | Best For |
|---------|--------------|-------------------|----------|
| **libSQL** (default) | `go build ./cmd/sgrep` | ~5-10 KB/vector | Large repos, production |
| sqlite-vec | `go build -tags=sqlite_vec ./cmd/sgrep` | ~780 KB/vector | Development, compatibility |

**libSQL advantages:**
- Uses DiskANN for approximate nearest neighbor search
- 93-177x more space-efficient than sqlite-vec
- Native F32_BLOB column type for vectors
- Compress neighbors option for index compression

## Commands

| Command | Description |
|---------|-------------|
| `sgrep [query]` | Semantic search (default) |
| `sgrep index [path]` | Index a directory |
| `sgrep watch [path]` | Watch and auto-index |
| `sgrep list` | List all indexed repos |
| `sgrep status` | Show index status |
| `sgrep clear` | Clear index |
| `sgrep setup` | Download embedding model, verify llama-server |
| `sgrep setup --with-rerank` | Also download reranker model (~636MB) |
| `sgrep server start` | Manually start embedding server |
| `sgrep server stop` | Stop embedding server |
| `sgrep server status` | Show server status |
| `sgrep install-claude-code` | Install Claude Code plugin |

## Claude Code Integration

Install the sgrep plugin for Claude Code with one command:

```bash
sgrep install-claude-code
```

This creates a plugin at `~/.claude/plugins/sgrep` that:
- **Auto-indexes** your project when Claude Code starts
- **Watch mode** keeps the index updated as you code
- **Skill documentation** teaches Claude when to use sgrep vs ripgrep

After installation, restart Claude Code to activate. The plugin works automatically—Claude will use sgrep for semantic searches like "how does authentication work" while using ripgrep for exact matches.

## Flags

| Flag | Description |
|------|-------------|
| `-n, --limit N` | Max results (default: 10) |
| `-c, --context` | Show code context |
| `--json` | JSON output for agents |
| `-q, --quiet` | Minimal output (paths only) |
| `--threshold F` | L2 distance threshold (default: 1.5, lower = stricter) |
| `-t, --include-tests` | Include test files in results (excluded by default) |
| `--all-chunks` | Show all matching chunks (disable deduplication) |
| `--hybrid` | Enable hybrid search (semantic + BM25) |
| `--semantic-weight F` | Weight for semantic score in hybrid mode (default: 0.6) |
| `--bm25-weight F` | Weight for BM25 score in hybrid mode (default: 0.4) |
| `--rerank` | Enable two-stage retrieval with cross-encoder reranking |
| `-d, --debug` | Show debug timing information |

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
4. **Storage**: Vectors stored in libSQL with DiskANN indexing
5. **Search**: Query embedded → DiskANN approximate nearest neighbor → load matching documents

**Smart skip for large repos**: When indexing repos with >1000 files, sgrep automatically filters out test files, generated code (*.pb.go, *.generated.go), and vendored directories to speed up indexing.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         sgrep                                │
├──────────────────────────────────────────────────────────────┤
│  Query: "error handling"                                     │
│         ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ llama.cpp   │───▶│  DiskANN    │───▶│   libSQL    │      │
│  │ Embedding   │    │   Search    │    │  Documents  │      │
│  │   (~15ms)   │    │   (~5ms)    │    │   (~5ms)    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│       ▲                    │                                 │
│       │                    ▼ (with --rerank)                 │
│       │              ┌─────────────┐                         │
│       │              │Cross-Encoder│                         │
│       │              │  Reranker   │                         │
│       │              │ (~300-700ms)│                         │
│       │              └─────────────┘                         │
│       │                                                      │
│       │ Auto-started by sgrep (16 parallel slots)           │
│       │ (daemon mode, continuous batching)                  │
│                                                              │
│  Total: ~30ms (or ~500ms with reranking)                    │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid Search Architecture

When `--hybrid` is enabled, sgrep combines semantic and lexical search:

```
Query: "authentication middleware"
         ↓
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  ┌─────────────┐         ┌─────────────┐           │
  │  │  Semantic   │         │    BM25     │           │
  │  │  (Vectors)  │         │   (FTS5)    │           │
  │  │    60%      │         │    40%      │           │
  │  └──────┬──────┘         └──────┬──────┘           │
  │         │                       │                   │
  │         └───────┬───────────────┘                   │
  │                 ↓                                   │
  │         ┌─────────────┐                            │
  │         │   Hybrid    │                            │
  │         │   Ranking   │                            │
  │         └─────────────┘                            │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

- **Semantic**: Understands intent ("auth" matches "authentication", "login", "session")
- **BM25**: Exact term matching with TF-IDF weighting (boosts exact "authentication" matches)

## Performance

Benchmarked on maestro codebase (102 files, 1572 chunks, 768-dim vectors):

| Metric | sgrep | ripgrep | 
|--------|-------|---------|
| Latency (avg) | **31ms** | 10ms |
| Token usage | **57% less** | baseline |
| Attempts needed | 1 | 3-7 |

**Embedding server optimization:**

The llama.cpp server is configured for maximum throughput:
- 16 parallel slots with continuous batching (`-cb`)
- Dynamic thread count based on CPU cores
- GPU acceleration (Metal on Mac, CUDA on Linux)

## Chunk Size Limits

The embedding model (nomic-embed-text) has a 2048 token context limit. sgrep handles this by:

1. Default chunk size: 1000 tokens (with AST-aware splitting)
2. Safety truncation at 1500 tokens in embedder
3. Large functions/types split into parts automatically

## Library Usage

Use sgrep as an embedded library in your Go application:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/sgrep"
)

func main() {
    ctx := context.Background()
    
    // Create client for a codebase
    client, err := sgrep.New("/path/to/codebase")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Index the codebase (required before searching)
    if err := client.Index(ctx); err != nil {
        log.Fatal(err)
    }

    // Search for code by semantic intent
    results, err := client.Search(ctx, "authentication logic", 10)
    if err != nil {
        log.Fatal(err)
    }

    for _, r := range results {
        fmt.Printf("%s:%d-%d (score: %.2f)\n", r.FilePath, r.StartLine, r.EndLine, r.Score)
    }
}
```

For more control, use the `pkg/` subpackages directly:
- `pkg/index` - Indexing and file watching
- `pkg/search` - Search with caching
- `pkg/embed` - Embedding generation
- `pkg/store` - Vector storage
- `pkg/chunk` - Code chunking with AST awareness

## License

Apache-2.0
