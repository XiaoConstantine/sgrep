#!/bin/bash
# Benchmark comparison: sgrep vs osgrep
# Run from the sgrep repo root

set -e

REPO_PATH="${1:-.}"
QUERIES=(
    "error handling"
    "authentication logic"
    "database connection"
    "API rate limiting"
    "user validation"
)

echo "=== Semantic Code Search Benchmark ==="
echo "Repository: $REPO_PATH"
echo "Date: $(date)"
echo ""

# Check tools are available
command -v sgrep >/dev/null 2>&1 || { echo "sgrep not found. Run: go install github.com/XiaoConstantine/sgrep@latest"; exit 1; }
command -v osgrep >/dev/null 2>&1 || { echo "osgrep not found. Run: npm install -g osgrep"; exit 1; }

cd "$REPO_PATH"

echo "=== Indexing Phase ==="
echo ""

# Index with sgrep
echo "Indexing with sgrep..."
SGREP_INDEX_START=$(python3 -c 'import time; print(time.time())')
sgrep index . 2>&1 | tail -3
SGREP_INDEX_END=$(python3 -c 'import time; print(time.time())')
SGREP_INDEX_TIME=$(python3 -c "print(f'{$SGREP_INDEX_END - $SGREP_INDEX_START:.2f}')")
echo "sgrep index time: ${SGREP_INDEX_TIME}s"
echo ""

# Index with osgrep
echo "Indexing with osgrep..."
OSGREP_INDEX_START=$(python3 -c 'import time; print(time.time())')
osgrep index 2>&1 | tail -3
OSGREP_INDEX_END=$(python3 -c 'import time; print(time.time())')
OSGREP_INDEX_TIME=$(python3 -c "print(f'{$OSGREP_INDEX_END - $OSGREP_INDEX_START:.2f}')")
echo "osgrep index time: ${OSGREP_INDEX_TIME}s"
echo ""

echo "=== Search Phase ==="
echo ""

# Warm up
sgrep "test" -n 1 > /dev/null 2>&1 || true
osgrep "test" -m 1 > /dev/null 2>&1 || true

SGREP_TOTAL=0
OSGREP_TOTAL=0

for query in "${QUERIES[@]}"; do
    echo "Query: \"$query\""
    
    # sgrep timing
    SGREP_START=$(python3 -c 'import time; print(time.time())')
    sgrep "$query" -n 5 > /dev/null 2>&1 || true
    SGREP_END=$(python3 -c 'import time; print(time.time())')
    SGREP_TIME=$(python3 -c "print(f'{($SGREP_END - $SGREP_START) * 1000:.0f}')")
    SGREP_TOTAL=$(python3 -c "print($SGREP_TOTAL + $SGREP_END - $SGREP_START)")
    
    # osgrep timing
    OSGREP_START=$(python3 -c 'import time; print(time.time())')
    osgrep "$query" -m 5 > /dev/null 2>&1 || true
    OSGREP_END=$(python3 -c 'import time; print(time.time())')
    OSGREP_TIME=$(python3 -c "print(f'{($OSGREP_END - $OSGREP_START) * 1000:.0f}')")
    OSGREP_TOTAL=$(python3 -c "print($OSGREP_TOTAL + $OSGREP_END - $OSGREP_START)")
    
    echo "  sgrep: ${SGREP_TIME}ms | osgrep: ${OSGREP_TIME}ms"
done

echo ""
echo "=== Summary ==="
echo ""
echo "| Metric | sgrep | osgrep |"
echo "|--------|-------|--------|"
echo "| Index time | ${SGREP_INDEX_TIME}s | ${OSGREP_INDEX_TIME}s |"

SGREP_AVG=$(python3 -c "print(f'{$SGREP_TOTAL / ${#QUERIES[@]} * 1000:.0f}')")
OSGREP_AVG=$(python3 -c "print(f'{$OSGREP_TOTAL / ${#QUERIES[@]} * 1000:.0f}')")
echo "| Avg search | ${SGREP_AVG}ms | ${OSGREP_AVG}ms |"

# Model info
echo ""
echo "=== Tool Details ==="
echo "sgrep: llama.cpp + nomic-embed-text-v1.5 (768d, local)"
echo "osgrep: transformers.js + mxbai-embed-xsmall-v1 (384d, local)"
