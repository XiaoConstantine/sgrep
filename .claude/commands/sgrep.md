# sgrep - Semantic Code Search

Use `sgrep` for semantic/conceptual code search. It understands intent, not just exact strings.

## When to Use

- Finding code by **concept**: "error handling", "authentication logic", "rate limiting"
- Exploring unfamiliar codebases
- When ripgrep patterns keep missing relevant code

## Quick Reference

```bash
# First time only
sgrep setup

# Index current directory
sgrep index .

# Search by intent
sgrep "database connection pooling"
sgrep "how are errors handled"
sgrep "JWT validation"

# With code context
sgrep -c "authentication middleware"

# JSON output (for parsing)
sgrep --json "error handling"

# Quiet mode (paths only)
sgrep -q "logging"
```

## Search Hierarchy

1. **sgrep** → Find files/functions by intent
2. **ast-grep** → Match structural patterns
3. **ripgrep** → Exact text search

## Example Workflow

```bash
# Find authentication code semantically
sgrep "user authentication flow"
# → auth/handler.go:45-80

# Then use ast-grep for structural patterns
sg -p 'if err != nil { return $_ }'

# Then ripgrep for specific symbols
rg "JWT_SECRET"
```

## Server Management

The embedding server auto-starts. Manual control:

```bash
sgrep server status   # Check status
sgrep server stop     # Stop server
```
