# sgrep - Agent Integration Guide

## Build & Test

```bash
# Build
go build -o sgrep ./cmd/sgrep

# Test
go test ./...

# Install
go install ./cmd/sgrep
```

## Prerequisites

llama.cpp server with embedding model:
```bash
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --port 8080
```

## Usage for Coding Agents

sgrep is designed to complement `ripgrep` and `ast-grep`:

| Tool | Use Case | Example |
|------|----------|---------|
| sgrep | Find by intent | `sgrep "authentication logic"` |
| ast-grep | Structural patterns | `sg -p '$fn($ctx, $err)'` |
| ripgrep | Exact strings | `rg "JWT_SECRET"` |

### Recommended Workflow

1. **Semantic discovery** → Find relevant files/functions
2. **Structural search** → Match patterns in those files
3. **Exact search** → Find specific symbols

### Output Modes

```bash
# Minimal (for token efficiency)
sgrep -q "error handling"
# → auth/handler.go:45-67

# With context
sgrep -c "error handling"
# → auth/handler.go:45-67
#     func handleError(err error) {
#       ...

# JSON (for programmatic use)
sgrep --json "error handling"
# → [{"file":"auth/handler.go","start":45,"end":67,"score":0.12}]
```

## Configuration

```bash
SGREP_ENDPOINT=http://localhost:8080  # llama.cpp server
SGREP_DIMS=768                        # embedding dimensions
SGREP_MAX_TOKENS=1500                 # chunk size
```

## Code Style

- Single package per directory
- No external LLM dependencies (local llama.cpp only)
- Minimal output by default (token-efficient)
- JSON output for agent parsing
