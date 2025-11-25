# sgrep - Agent Integration Guide

## Build & Test

```bash
# Build
go build -o sgrep ./cmd/sgrep

# Test
go test ./...

# Lint
golangci-lint run ./...

# Install
go install ./cmd/sgrep
```

## Setup

```bash
# Install llama.cpp (macOS)
brew install llama.cpp

# Download embedding model and verify installation
sgrep setup
```

The embedding server auto-starts when needed. No manual server management required.

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

## Server Management

The embedding server runs as a daemon (auto-started, stays running):

```bash
sgrep server status   # Check if running
sgrep server stop     # Stop daemon
sgrep server start    # Manually start
```

## Configuration

```bash
SGREP_HOME=~/.sgrep                   # Data directory
SGREP_ENDPOINT=http://localhost:8080  # Override server URL
SGREP_PORT=8080                       # Server port
SGREP_DIMS=768                        # Embedding dimensions
```

## Code Style

- Single package per directory
- No external LLM dependencies (local llama.cpp only)
- Minimal output by default (token-efficient)
- JSON output for agent parsing
- Use `_ = fn()` for intentionally ignored errors
