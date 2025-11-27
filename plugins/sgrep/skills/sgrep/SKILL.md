# sgrep - Smart Code Search

**sgrep** is a semantic and hybrid code search tool that understands what you mean, not just what you type.

- **License:** Apache 2.0
- **Repository:** https://github.com/XiaoConstantine/sgrep

## When to Use

Use `sgrep` instead of grep/ripgrep when:
- Searching by **concept** or **intent** ("how does authentication work")
- Looking for code with **specific terms + context** (use `--hybrid`)
- Exploring unfamiliar codebases
- grep patterns keep missing relevant code

## Commands

```bash
# Semantic search (understands intent)
sgrep "error handling logic"
sgrep "database connection pooling"

# Hybrid search (semantic + exact term matching)
sgrep --hybrid "JWT token validation"
sgrep --hybrid "OAuth2 refresh"

# With code context
sgrep -c "authentication middleware"

# JSON output
sgrep --json "rate limiting"

# Quiet mode (paths only)  
sgrep -q "logging"
```

## Semantic vs Hybrid

| Mode | Best For | Example |
|------|----------|---------|
| Default (semantic) | Conceptual queries | "how does caching work" |
| `--hybrid` | Queries with specific terms | "parseConfig function" |

**Use `--hybrid`** when your query contains function names, API names, or technical terms that should match exactly.

## Search Hierarchy

1. **sgrep** → Find files by semantic intent
2. **sgrep --hybrid** → Find files by intent + specific terms  
3. **ast-grep** → Match structural patterns
4. **ripgrep** → Exact text search

## Tips

- Frame queries as questions: "how are errors handled"
- Use `--hybrid` for specific function/class names
- Use `-c` to see code snippets in results
- Use `--json` for programmatic parsing
- Results show `file:startLine-endLine` format
