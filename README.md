# sgrep

**Local semantic and hybrid search for codebases and coding-agent history.**

Find implementation by intent, then recover the decisions and fixes that led
to it across Claude Code, Codex, Cursor, OpenCode, and Pi sessions.

[Documentation](https://xiaocui.me/sgrep/) ·
[Getting started](https://xiaocui.me/sgrep/guides/getting-started/) ·
[CLI reference](https://xiaocui.me/sgrep/reference/cli/) ·
[Agent skill](SKILL.md)

## Why sgrep?

- Search code by meaning instead of guessing identifiers.
- Search prior agent sessions without reopening transcripts one by one.
- Keep source, conversations, models, and indexes on your machine.
- Give coding agents compact file-and-line results instead of large context dumps.

```bash
sgrep "where do we validate authentication tokens"
sgrep conv "why did we change the authentication cache key"
```

## Install

```bash
brew tap XiaoConstantine/tap
brew install sgrep

# Download the local embedding model and verify llama.cpp.
sgrep setup
```

See the [installation guide](https://xiaocui.me/sgrep/guides/getting-started/)
for curl, Go, source builds, and sqlite-vec.

## Search a Codebase

```bash
cd your-repository
sgrep index .

sgrep "database connection error handling"              # balanced, the default
sgrep --profile fast "request routing"                   # lowest latency
sgrep --profile quality "JWT validation middleware"     # best ranking
```

| Profile | Pipeline | Use it for |
|---------|----------|------------|
| `fast` | Semantic | Quick exploration |
| `balanced` | Semantic + BM25 | Everyday search; default |
| `quality` | Semantic + BM25 + ColBERT | Harder ranking problems |

[Code-search guide →](https://xiaocui.me/sgrep/guides/code-search/)

## Search Agent Conversations

```bash
sgrep conv index

sgrep conv "embedding server ownership bug"
sgrep conv "session compaction" --agent codex --since 7d
sgrep conv resume <session_id>
```

Conversation search supports Claude Code, Codex CLI, Cursor, OpenCode, and Pi,
with semantic, hybrid, and exact retrieval.

[Conversation-search guide →](https://xiaocui.me/sgrep/guides/conversation-search/)

## Built for Coding Agents

`sgrep` complements structural and exact search:

| Tool | Finds | Example |
|------|-------|---------|
| `sgrep` | Intent | `sgrep "authentication logic"` |
| `ast-grep` | Structure | `sg -p '$fn($args)'` |
| `ripgrep` | Exact text | `rg "JWT_SECRET"` |

Install the shared agent skill:

```bash
npx skills add XiaoConstantine/sgrep --skill sgrep -g
```

Claude Code users can also install automatic project indexing and watch hooks:

```bash
sgrep install-claude-code
```

[Agent-integration guide →](https://xiaocui.me/sgrep/guides/agent-integration/)

## Architecture

[![sgrep architecture: code and conversation sources feed a shared local embedding service and separate retrieval pipelines](docs/static/architecture.jpg)](docs/static/architecture.jpg)

The managed local llama.cpp service is shared, while code and conversation
indexes retain separate retrieval controls. The editable diagram source is at
[docs/static/architecture.tldr](docs/static/architecture.tldr).

[Architecture details →](https://xiaocui.me/sgrep/architecture/)

## Current Benchmark

Sequential top-10 search over `dspy-go` at commit
`87cb50fa0611ef8a3905494c8b16fb0aab7666d3`: 543 files, 13,136 chunks,
61,082 ColBERT segments, and 20 judged queries on an Apple M3 Pro.

| Profile | MRR | Recall@5 | Mean | p95 |
|---------|----:|---------:|-----:|----:|
| `fast` | 0.470 | 0.300 | 30.2ms | 37ms |
| `balanced` | 0.593 | 0.350 | 35.9ms | 44ms |
| `quality` | **0.729** | **0.408** | 46.5ms | 56ms |

These are small after-state smoke results, not a controlled pre/post study.
See the [benchmark methodology and caveats](https://xiaocui.me/sgrep/benchmarks/).

## Documentation

| Guide | Covers |
|-------|--------|
| [Getting started](https://xiaocui.me/sgrep/guides/getting-started/) | Installation, setup, first searches |
| [Code search](https://xiaocui.me/sgrep/guides/code-search/) | Profiles, output, watch mode, reranking |
| [Conversation search](https://xiaocui.me/sgrep/guides/conversation-search/) | Sources, filters, view, export, resume |
| [CLI reference](https://xiaocui.me/sgrep/reference/cli/) | Commands and flags |
| [Configuration](https://xiaocui.me/sgrep/reference/configuration/) | Environment and server modes |
| [Storage](https://xiaocui.me/sgrep/reference/storage/) | Index files and vector backends |
| [Library API](https://xiaocui.me/sgrep/library/) | Embedding sgrep in Go |

## Development

```bash
go build -o sgrep ./cmd/sgrep
go test ./...
golangci-lint run ./...
```

## License

Apache-2.0
