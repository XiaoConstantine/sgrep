# sgrep Benchmark

Compare semantic search (sgrep) vs lexical search (grep/ripgrep) for agent efficiency.

## Metrics

| Metric | Description |
|--------|-------------|
| **Tokens Used** | Total tokens in queries + results |
| **Attempts** | Number of search attempts to find target |
| **Precision** | % of results that are relevant |
| **Recall** | % of relevant code found |
| **Latency** | Time to first result |

## Test Cases

Real-world agent queries that are hard for grep:

```yaml
queries:
  - intent: "How does authentication work?"
    target_files: ["auth/middleware.go", "auth/jwt.go"]
    grep_patterns: ["auth", "authenticate", "login", "session", "token", "jwt"]
    
  - intent: "Where are database connections managed?"
    target_files: ["db/pool.go", "db/connect.go"]
    grep_patterns: ["database", "db", "sql", "connection", "pool"]
    
  - intent: "How are errors handled in the API layer?"
    target_files: ["api/errors.go", "handlers/error.go"]
    grep_patterns: ["error", "Error", "err", "handle", "response"]
    
  - intent: "What's the caching strategy?"
    target_files: ["cache/redis.go", "cache/memory.go"]
    grep_patterns: ["cache", "Cache", "redis", "ttl", "expire"]
```

## Running the Benchmark

```bash
# 1. Index a real codebase
sgrep index /path/to/codebase

# 2. Run benchmark
go run ./benchmark/main.go /path/to/codebase

# 3. Compare results
cat benchmark_results.json
```

## Expected Results

Based on mgrep's findings, semantic search should show:
- **~2x fewer tokens** (one query vs multiple grep attempts)
- **Higher first-try accuracy** (semantic understanding vs pattern guessing)
- **Slightly higher latency** (embedding generation overhead)

## Token Counting

```
grep approach:
  query1: "rg authenticate" → 50 tokens output, no match
  query2: "rg 'auth.*handler'" → 200 tokens output, partial
  query3: "rg 'func.*Auth'" → 150 tokens output, found it
  Total: 3 queries × ~10 tokens + 400 tokens output = 430 tokens

sgrep approach:
  query1: "sgrep 'authentication logic'" → 100 tokens output, found it
  Total: 1 query × ~5 tokens + 100 tokens output = 105 tokens

Savings: 4x fewer tokens
```
