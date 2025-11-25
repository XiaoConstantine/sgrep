#!/bin/bash
# Quick benchmark comparing sgrep vs ripgrep

set -e

CODEBASE="${1:-.}"
QUERY="${2:-error handling}"

echo "=== Benchmark: sgrep vs ripgrep ==="
echo "Codebase: $CODEBASE"
echo "Query: $QUERY"
echo ""

# Ensure sgrep is indexed
if ! sgrep status > /dev/null 2>&1; then
    echo "Indexing codebase first..."
    sgrep index "$CODEBASE"
    echo ""
fi

# Measure sgrep
echo "--- sgrep (semantic) ---"
SGREP_START=$(python3 -c "import time; print(int(time.time()*1000))")
SGREP_OUTPUT=$(sgrep -n 5 "$QUERY" 2>&1 || true)
SGREP_END=$(python3 -c "import time; print(int(time.time()*1000))")
SGREP_LATENCY=$((SGREP_END - SGREP_START))
SGREP_LINES=$(echo "$SGREP_OUTPUT" | wc -l)
SGREP_TOKENS=$(echo "$SGREP_OUTPUT" | wc -w)

echo "$SGREP_OUTPUT"
echo ""
echo "Latency: ${SGREP_LATENCY}ms"
echo "Output lines: $SGREP_LINES"
echo "Approx tokens: $((SGREP_TOKENS * 13 / 10))"
echo ""

# Measure ripgrep with multiple patterns (simulating agent behavior)
echo "--- ripgrep (lexical, multiple attempts) ---"
PATTERNS=($(echo "$QUERY" | tr ' ' '\n'))
RG_TOTAL_TOKENS=0
RG_ATTEMPTS=0
RG_START=$(python3 -c "import time; print(int(time.time()*1000))")

for pattern in "${PATTERNS[@]}"; do
    RG_ATTEMPTS=$((RG_ATTEMPTS + 1))
    RG_OUTPUT=$(rg -l --max-count 5 "$pattern" "$CODEBASE" 2>/dev/null || true)
    RG_WORDS=$(echo "$RG_OUTPUT" | wc -w)
    RG_TOTAL_TOKENS=$((RG_TOTAL_TOKENS + RG_WORDS * 13 / 10))
    
    if [ -n "$RG_OUTPUT" ]; then
        echo "Pattern '$pattern': $(echo "$RG_OUTPUT" | wc -l) files"
    fi
done

RG_END=$(python3 -c "import time; print(int(time.time()*1000))")
RG_LATENCY=$((RG_END - RG_START))

echo ""
echo "Attempts: $RG_ATTEMPTS"
echo "Latency: ${RG_LATENCY}ms"
echo "Approx tokens: $RG_TOTAL_TOKENS"
echo ""

# Summary
echo "=== Summary ==="
if [ "$SGREP_TOKENS" -lt "$RG_TOTAL_TOKENS" ]; then
    SAVINGS=$(( (RG_TOTAL_TOKENS - SGREP_TOKENS) * 100 / RG_TOTAL_TOKENS ))
    echo "Winner: sgrep ($SAVINGS% fewer tokens)"
else
    echo "Winner: ripgrep"
fi
