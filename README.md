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

# Default build (libSQL source index + compact TQ-MSE first-stage search)
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

# Index your codebase (TQ-MSE chunk/file vectors + ColBERT preindex, auto-starts embedding server)
sgrep index .

# Optional: override the default TQ-MSE ColBERT codec
sgrep index . --colbert-codec tqmse
sgrep index . --colbert-codec int8
sgrep index . --colbert-codec pq6

# Optional: keep full SQL vectors for legacy vector search
sgrep index . --sql-vectors

# Balanced profile (default): semantic + lexical retrieval
sgrep "error handling for database connections"

# Explicit speed and quality profiles
sgrep --profile fast "quick semantic exploration"
sgrep --profile quality "JWT token validation logic"
sgrep --profile quality "how are API rate limits implemented"

# Quality profile with custom RRF signal weights
sgrep --profile quality "authentication middleware" --semantic-weight 0.5 --bm25-weight 0.5

# Watch mode (background indexing)
sgrep watch .
```

The embedding server starts automatically when needed and stays running as a daemon.

## Conversation Search

Search across conversations from Claude Code, Codex CLI, Cursor, OpenCode, and Pi.

```bash
# Index conversations (auto-starts embedding server)
sgrep conv index

# Index a single agent
sgrep conv index --source claude
sgrep conv index --source codex
sgrep conv index --source cursor
sgrep conv index --source opencode
sgrep conv index --source pi

# Watch mode (auto-index new conversations)
sgrep conv index --watch

# Re-index all cached sessions
sgrep conv index --force

# Search conversations
sgrep conv "authentication"
sgrep conv search "authentication" --agent codex --limit 5
sgrep conv search "session compaction" --agent pi --limit 5
sgrep conv "JWT token" --hybrid
sgrep conv "database migration" --agent claude --since 7d
sgrep conv "auth" --exact
sgrep conv "bug fix" --project payment-service --after 2026-01-01 --before 2026-06-01

# View, export, or resume a session
sgrep conv view <session_id>
sgrep conv view <session_id> --turn 3 --no-color
sgrep conv export <session_id> -o conversation.md
sgrep conv export <session_id> --format json -o conversation.json
sgrep conv resume <session_id>

# Extract context for injection into new session
sgrep conv context <session_id>
sgrep conv context <session_id> --turns 10 --copy

# Copy to clipboard
sgrep conv copy <session_id>
sgrep conv copy <session_id> --turn 2 --code-only

# Check index status
sgrep conv status
```

**Watch mode** monitors conversation directories for all agents and automatically indexes new sessions as they're created. This ensures your conversation search stays up-to-date without manual re-indexing.

Conversations are stored at `~/.sgrep/conversations/conv.db` with compact
TQ-MSE turn vectors in `~/.sgrep/conversations/turn_embeddings.tqmse`.
Re-running `sgrep conv index` backfills missing embeddings for existing
sessions and refreshes the compact conversation vector sidecar.

Default search output includes the agent session/conversation ID so you can jump
straight to `view`, `resume`, or external agent tooling with the exact session
identifier.

## Hybrid Search

Hybrid search combines **semantic understanding** with **lexical matching (BM25)** for improved accuracy. This helps when:
- Searching for specific technical terms (e.g., "JWT", "OAuth", "mutex")
- The query contains exact function/variable names
- Semantic search alone misses exact keyword matches

```bash
# Default balanced profile: semantic + BM25 fused with weighted RRF
sgrep "authentication"

# Equivalent explicit profile
sgrep --profile balanced "authentication"

# Custom weights: more emphasis on exact matches
sgrep --hybrid --semantic-weight 0.4 --bm25-weight 0.6 "parseAST"
```

**Note**: Hybrid search requires building with FTS5 support (see [From Source](#from-source)). The FTS5 index is created automatically on first hybrid search - no re-indexing needed.

## Multi-Stage Retrieval Pipeline

sgrep uses a sophisticated multi-stage retrieval pipeline for maximum accuracy:

```
Query: "authentication middleware"
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Hybrid Retrieval (--hybrid)                            │
│ ┌───────────────┐    ┌───────────────┐                         │
│ │   Semantic    │    │     BM25      │                         │
│ │  (TQ-MSE)     │    │    (FTS5)     │                         │
│ │     60%       │    │     40%       │                         │
│ └───────┬───────┘    └───────┬───────┘                         │
│         └────────┬───────────┘                                  │
│                  ↓                                              │
│         Top 50 candidates                                       │
└─────────────────────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: ColBERT Late Interaction (--colbert)                   │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │  Token-level similarity: MaxSim(query_tokens, doc_tokens) │  │
│ │  Scores all 50 candidates with fine-grained matching      │  │
│ └───────────────────────────────────────────────────────────┘  │
│                  ↓                                              │
│         Re-scored candidates                                    │
└─────────────────────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Cross-Encoder Reranking (--rerank)                     │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │  Full attention: query ⊗ document → relevance score       │  │
│ │  Reranks the top ColBERT candidates                       │  │
│ └───────────────────────────────────────────────────────────┘  │
│                  ↓                                              │
│         Final ranked results                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval Modes

| Mode | Command | Best For | Notes |
|------|---------|----------|-------|
| Fast | `sgrep --profile fast "query"` | Lowest latency | Semantic retrieval only |
| Balanced | `sgrep --profile balanced "query"` | Default agent search | Semantic + lexical weighted RRF |
| **Quality** | `sgrep --profile quality "query"` | **Best code accuracy** | Adds precomputed late interaction |
| Cascade | `sgrep --profile quality --rerank "query"` | Experiments / non-code text | Generic cross-encoder remains off by default |

**Recommended for code**: use the default balanced profile for exploration and `--profile quality` when ranking quality matters most.

Profiles are forced rather than inferred from artifacts, making quality and latency benchmarks reproducible.

```bash
sgrep --profile quality "authentication middleware"
sgrep --profile fast "error handling"
sgrep --profile quality --semantic-weight 0.5 --bm25-weight 0.5 "JWT token"
```

### Setup

```bash
# Basic setup (embedding model only, ~130MB)
sgrep setup

# With cross-encoder reranking (~636MB additional)
sgrep setup --with-rerank
```

**Note**: ColBERT scoring uses the same embedding model—no additional setup required. Cross-encoder reranking requires a separate model download.

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
│   │   ├── index.db              # libSQL metadata + source chunks + FTS
│   │   ├── vectors.tqmse         # Compact TQ-MSE chunk vector artifact
│   │   ├── vectors.mmap          # Exact float32 artifact for small/medium indexes
│   │   ├── file_vectors.tqmse    # Compact TQ-MSE file/document vector artifact
│   │   ├── colbert_segments.mmap # Precomputed ColBERT segments (when enabled)
│   │   └── metadata.json         # Repo path, index time
│   └── d4e5f6/              # Hash of /path/to/repo2
│       └── ...
├── conversations/
│   ├── conv.db              # Conversation metadata, embeddings, and FTS
│   └── turn_embeddings.tqmse # Compact TQ-MSE conversation turn vectors
├── server.pid               # Embedding server PID
└── server.log               # Embedding server logs
```

Use `sgrep list` to see all indexed repositories.

## Storage Backends

sgrep supports two SQL metadata backends. Full indexing writes compact
`vectors.tqmse` and `file_vectors.tqmse` artifacts. Indexes up to 20,000 chunks
also receive a resident exact-float32 `vectors.mmap` artifact; larger indexes
use the lower-memory TQ-MSE scan. SQLite/libSQL remains the source of truth for
chunk content, metadata, and FTS/BM25.

| Backend | Build Command | Storage Efficiency | Best For |
|---------|--------------|-------------------|----------|
| **libSQL** (default) | `go build ./cmd/sgrep` | Compact TQ-MSE retrieval + libSQL metadata | Large repos, production |
| sqlite-vec | `go build -tags=sqlite_vec ./cmd/sgrep` | Simpler fallback backend | Development, compatibility |

**Default build advantages:**
- Prefers exact float32 scanning for indexes up to `--exact-vector-limit` (20,000 chunks by default)
- Falls back to compact `vectors.tqmse` scanning above that limit
- Uses `file_vectors.tqmse` for compact document/file-level semantic retrieval
- Avoids full SQL vector blobs during normal full indexing
- Keeps libSQL as the source of truth for chunks, metadata, and FTS/BM25
- Keeps the legacy SQL vector path available with `sgrep index --sql-vectors`
- Falls back to SQL vector search only when compact artifacts are missing and SQL vectors exist

`sgrep watch` keeps SQL vectors as its incremental-update backing store so it
can refresh compact TQ-MSE sidecars after file changes. Run `sgrep index` again
when you want to re-compact the index after a long watch session.

## Commands

| Command | Description |
|---------|-------------|
| `sgrep [query]` | Balanced semantic + lexical search (default) |
| `sgrep --profile fast/balanced/quality [query]` | Force a reproducible retrieval profile |
| `sgrep index [path]` | Index a directory |
| `sgrep index [path] --exact-vector-limit N` | Set exact float32 cutoff; 0 disables |
| `sgrep index [path] --colbert-codec tqmse/int8/pq6` | Select ColBERT segment codec |
| `sgrep index [path] --sql-vectors` | Also store full SQL vectors for legacy search |
| `sgrep watch [path]` | Watch and auto-index |
| `sgrep list` | List all indexed repos |
| `sgrep status` | Show index status |
| `sgrep clear` | Clear index |
| `sgrep conv [search] <query>` | Search indexed agent conversations |
| `sgrep conv index [--source agent] [--watch] [--force]` | Index or watch conversations |
| `sgrep conv view <session_id> [--turn N] [--no-color]` | View a conversation |
| `sgrep conv export <session_id> --format markdown/json/html` | Export a conversation |
| `sgrep conv context <session_id> [--turns N] [--copy]` | Extract context for a new session |
| `sgrep conv copy <session_id> [--turn N] [--code-only]` | Copy a conversation or turn |
| `sgrep conv status` | Show conversation index status |
| `sgrep setup` | Download embedding model, verify llama-server |
| `sgrep setup --with-rerank` | Also download reranker model (~636MB) |
| `sgrep server start` | Manually start embedding server |
| `sgrep server stop` | Stop embedding server |
| `sgrep server status` | Show server status |
| `sgrep install-claude-code` | Install Claude Code plugin + Claude skill |
| `sgrep install-skill` | Install offline fallback copies for Claude and `.agents` clients |

## Claude Code Integration

Install the sgrep plugin for Claude Code with one command:

```bash
sgrep install-claude-code
```

This creates:
- a plugin at `~/.claude/plugins/sgrep`
- a user-level Claude skill at `~/.claude/skills/sgrep/SKILL.md`

The installation:
- **Auto-indexes** your project when Claude Code starts
- **Watch mode** keeps the index updated as you code
- **Installs the standard Claude skill** that teaches Claude when to use sgrep vs ripgrep

After installation, restart Claude Code to activate. The plugin works automatically—Claude will use sgrep for semantic searches like "how does authentication work" while using ripgrep for exact matches.

## Shared Skill Installation

Install the reusable skill globally with the same cross-agent installer used by
skills.sh packages. This installs the agent instructions; install the `sgrep`
binary separately using the installation steps above.

```bash
npx skills add XiaoConstantine/sgrep --skill sgrep -g
```

For unattended installation, append `-y`. The installer discovers the root
`SKILL.md` and links it into the supported global agent directories. To target
one client, add an agent selector such as `--agent claude-code`.

If Node.js is unavailable, the binary retains an offline fallback:

```bash
sgrep install-skill
```

The fallback writes only `~/.agents/skills/sgrep/SKILL.md` and
`~/.claude/skills/sgrep/SKILL.md`; the `npx skills` command is preferred for
cross-client installation and updates.

## Agent Integration Standards

- **Open Agent Skill**: the canonical package entrypoint is the repository-root [`SKILL.md`](SKILL.md), matching skills.sh repository discovery.
- **Distribution copies**: `.agents/skills/sgrep/SKILL.md` and `plugins/sgrep/skills/sgrep/SKILL.md` are kept byte-for-byte identical by tests.
- **Codex / Amp**: use the repo's root [AGENTS.md](AGENTS.md) for shared workflow guidance.
- **Cross-client skills**: `npx skills` installs the canonical skill into each selected agent's supported global location.
- **Claude Code plugin**: `sgrep install-claude-code` additionally installs session hooks and a Claude-specific plugin copy.
- **Claude slash commands**: `.claude/commands/<name>.md` are explicit commands and complement skills rather than replacing them.

## Flags

| Flag | Description |
|------|-------------|
| `-n, --limit N` | Max results (default: 10) |
| `-c, --context` | Show code context |
| `--json` | JSON output for agents |
| `-q, --quiet` | Minimal output (paths only) |
| `--threshold F` | Cosine distance threshold (default: 1.5, lower = stricter) |
| `-t, --include-tests` | Include test files in results (excluded by default) |
| `--all-chunks` | Show all matching chunks (disable deduplication) |
| `--profile fast/balanced/quality` | Select an explicit retrieval pipeline (default: balanced) |
| `--hybrid` | Override profile hybrid retrieval |
| `--colbert` / `--no-colbert` | Override profile late interaction |
| `--semantic-weight F` | Weight for semantic score in hybrid mode (default: 0.6) |
| `--bm25-weight F` | Weight for BM25 score in hybrid mode (default: 0.4) |
| `--rerank` | Enable cross-encoder reranking (requires `sgrep setup --with-rerank`) |
| `-d, --debug` | Show debug timing information |

## Configuration

Environment variables:
```bash
SGREP_HOME=~/.sgrep                    # Data storage location
SGREP_PORT=8080                        # Managed embedding server port
SGREP_ENDPOINT=http://host:port        # External server mode (sgrep does not manage its process)
SGREP_CONTEXT_TOKENS=512               # Per-slot context; changing it requires reindexing
SGREP_DIMS=768                         # Vector dimensions
SGREP_VECTOR_BACKEND=tqmse             # Force TQ-MSE search
SGREP_VECTOR_BACKEND=libsql            # Force legacy SQL vector search (requires --sql-vectors index)
SGREP_CONV_VECTOR_BACKEND=tqmse        # Force TQ-MSE conversation vector search
SGREP_CONV_VECTOR_BACKEND=sqlite       # Force legacy SQL conversation vector scan
```

## How It Works

1. **Setup**: `sgrep setup` downloads the embedding model and verifies llama-server
2. **Indexing**: Files are chunked using AST-aware splitting (Go, TS, Python) or size-based fallback
3. **Embedding**: Each chunk is embedded via llama.cpp (local, auto-started)
4. **Storage**: Chunk/file vectors are encoded into `vectors.tqmse` / `file_vectors.tqmse`; SQL stores chunk content, metadata, and FTS. `--sql-vectors` also keeps full SQL vectors.
5. **Search**: Query embedded → vector/hybrid retrieval → optional ColBERT late interaction → optional rerank

**Smart skip for large repos**: When indexing repos with >1000 files, sgrep automatically filters out test files, generated code (*.pb.go, *.generated.go), and vendored directories to speed up indexing.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         sgrep                                │
├──────────────────────────────────────────────────────────────┤
│  Query: "error handling"                                     │
│         ↓                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ llama.cpp   │───▶│ TQ-MSE      │───▶│   libSQL    │      │
│  │ Embedding   │    │ + BM25/FTS5 │    │ metadata    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│       ▲                    │                                 │
│       │                    ▼ (with --colbert)                │
│       │              ┌─────────────┐                         │
│       │              │  ColBERT    │                         │
│       │              │ Late-Interx │                         │
│       │              └──────┬──────┘                         │
│       │                     │                                │
│       │                     ▼ (with --rerank)                │
│       │              ┌─────────────┐                         │
│       │              │Cross-Encoder│                         │
│       │              │  Reranker   │                         │
│       │              └─────────────┘                         │
│       │                                                      │
│       │ Auto-started by sgrep                                │
│       │ (daemon mode, continuous batching)                   │
│                                                              │
│  Recommended: --hybrid --colbert                             │
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

## Recent Benchmarks

Current `dspy-go` benchmark on Apple M3 Pro + Metal at corpus commit
`87cb50fa0611ef8a3905494c8b16fb0aab7666d3` (543 files, 13,136 chunks,
61,082 ColBERT segments; 20 judged queries, sequential execution):

| Profile | MRR | Recall@5 | Mean latency | p95 latency |
|---------|----:|---------:|-------------:|------------:|
| `fast` | 0.470 | 0.300 | 30.2ms | 37ms |
| `balanced` | 0.593 | 0.350 | 35.9ms | 44ms |
| `quality` | **0.729** | **0.408** | 46.5ms | 56ms |

The base index took 3m41s and the complete index with default ColBERT
precomputation took 7m43s. Exact float32 top-50 scans measured about 0.63ms
for 10k vectors and 6.8ms for 100k vectors; TQ-MSE measured about 4.9ms and
49.3ms respectively. Full query logs are in
`bench/results/current-implementation/`.

Notes:
- Chunk and document-level semantic retrieval use TQ-MSE artifacts by default
  in the libSQL build; set `SGREP_VECTOR_BACKEND=libsql` to force the old
  libSQL vector path only for indexes built with `--sql-vectors`.
- The default ColBERT segment codec is `tqmse`, which targets about half the
  segment storage of int8 while keeping int8 available as a conservative
  `--colbert-codec int8` override.
- `--colbert-codec pq6` is size-gated: small repos automatically stay on int8.
- Search latency varies significantly with corpus size, hardware, and whether ColBERT or rerank are enabled, so the README avoids claiming one universal query-time number.

## Chunk Size Limits

The chunker, indexer, embedder, and managed llama.cpp server share one per-slot context budget (512 tokens by default):

1. Document chunks reserve 64 tokens for Nomic retrieval task prefixes and estimation variance.
2. Large functions/types are split at AST and line boundaries before embedding.
3. Oversized direct embedding requests return an error; source documents are never silently truncated.
4. Queries use `search_query:` and indexed chunks use `search_document:`. Changing the context or embedding format requires reindexing.

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
