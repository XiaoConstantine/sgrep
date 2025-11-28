#!/bin/bash
# Quality comparison: sgrep vs osgrep
# Compares search result QUALITY, not just speed
#
# Methodology:
# 1. Define queries with known "ground truth" expected files
# 2. Run both tools and capture actual results  
# 3. Calculate metrics: precision, recall, result overlap
# 4. Human-readable report with examples

set -e

REPO_PATH="${1:-/Users/xiao/development/github.com/XiaoConstantine/maestro}"
SGREP_BIN="${2:-./sgrep}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/benchmark/quality"

mkdir -p "$OUTPUT_DIR"

cd "$REPO_PATH"

echo "=== Semantic Search Quality Comparison ==="
echo "Repository: $REPO_PATH"
echo "Date: $(date)"
echo ""

# ========================================
# TEST QUERIES WITH GROUND TRUTH
# ========================================
# Format: "query|expected_file1,expected_file2,..."
# Expected files are files that SHOULD be in top-5 results

declare -a TEST_CASES=(
    # Query 1: Clear keyword match + semantic understanding
    "embedding generation|embedding_router.go,embedding_cache.go,multi_vector_embedding.go"
    
    # Query 2: Conceptual - needs semantic understanding
    "how to validate code reviews|review_processor.go,review.go,consensus_validation_processor.go"
    
    # Query 3: GitHub API operations
    "fetch pull request from github|github_tool.go,review.go"
    
    # Query 4: Error handling patterns
    "error handling and recovery|helper_functions.go,review_processor.go"
    
    # Query 5: RAG/vector search
    "store and retrieve vector embeddings|rag.go,embedding_router.go"
    
    # Query 6: AST parsing
    "parse go source code into functions|chunk.go,context_extractor.go"
    
    # Query 7: LLM orchestration  
    "coordinate multiple ai agents|agentic_orchestrator.go,agent_spawner.go,unified_react_agent.go"
    
    # Query 8: Session management
    "manage claude sessions|claude_session_manager.go,claude_coordinator.go"
    
    # Query 9: Comment processing
    "process and refine review comments|comment_processor.go,comment_refinement_processor.go"
    
    # Query 10: Similarity/deduplication
    "calculate text similarity and deduplicate|similarity_utils.go,result_aggregator.go"
)

# ========================================
# HELPER FUNCTIONS
# ========================================

extract_files_sgrep() {
    local query="$1"
    "$SGREP_BIN" "$query" --json -n 10 2>/dev/null | \
        jq -r '.[].file' 2>/dev/null | \
        xargs -I{} basename {} | \
        sort -u | head -10
}

extract_files_osgrep() {
    local query="$1"
    osgrep search "$query" --json -m 10 2>/dev/null | \
        jq -r '.results[].path' 2>/dev/null | \
        xargs -I{} basename {} | \
        sort -u | head -10
}

# Calculate overlap between two file lists
calculate_overlap() {
    local files1="$1"  # comma-separated
    local files2="$2"  # comma-separated
    
    # Convert to arrays
    IFS=',' read -ra arr1 <<< "$files1"
    IFS=',' read -ra arr2 <<< "$files2"
    
    local overlap=0
    for f1 in "${arr1[@]}"; do
        for f2 in "${arr2[@]}"; do
            if [[ "$f1" == "$f2" ]]; then
                ((overlap++))
                break
            fi
        done
    done
    
    echo "$overlap"
}

# Calculate precision: how many of returned results are in expected?
calculate_precision() {
    local returned="$1"  # newline-separated
    local expected="$2"  # comma-separated
    
    local returned_count=$(echo "$returned" | grep -c '.' || echo 0)
    [[ $returned_count -eq 0 ]] && echo "0" && return
    
    local hits=0
    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        if [[ "$expected" == *"$file"* ]]; then
            ((hits++))
        fi
    done <<< "$returned"
    
    python3 -c "print(f'{$hits / $returned_count:.2f}')"
}

# Calculate recall: how many of expected appear in results?
calculate_recall() {
    local returned="$1"  # newline-separated
    local expected="$2"  # comma-separated
    
    IFS=',' read -ra expected_arr <<< "$expected"
    local expected_count=${#expected_arr[@]}
    [[ $expected_count -eq 0 ]] && echo "0" && return
    
    local hits=0
    for exp in "${expected_arr[@]}"; do
        if echo "$returned" | grep -q "$exp"; then
            ((hits++))
        fi
    done
    
    python3 -c "print(f'{$hits / $expected_count:.2f}')"
}

# ========================================
# RUN COMPARISON
# ========================================

echo "Running ${#TEST_CASES[@]} test queries..."
echo ""

# Results arrays
declare -a SGREP_PRECISIONS
declare -a SGREP_RECALLS
declare -a OSGREP_PRECISIONS
declare -a OSGREP_RECALLS
declare -a OVERLAPS

RESULTS_FILE="$OUTPUT_DIR/detailed_results.md"
echo "# Quality Comparison Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "Repository: $REPO_PATH" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for test_case in "${TEST_CASES[@]}"; do
    IFS='|' read -r query expected <<< "$test_case"
    
    echo "Query: \"$query\""
    
    # Get results from both tools
    sgrep_files=$(extract_files_sgrep "$query")
    osgrep_files=$(extract_files_osgrep "$query")
    
    # Convert to comma-separated for comparison
    sgrep_csv=$(echo "$sgrep_files" | tr '\n' ',' | sed 's/,$//')
    osgrep_csv=$(echo "$osgrep_files" | tr '\n' ',' | sed 's/,$//')
    
    # Calculate metrics
    sgrep_precision=$(calculate_precision "$sgrep_files" "$expected")
    sgrep_recall=$(calculate_recall "$sgrep_files" "$expected")
    osgrep_precision=$(calculate_precision "$osgrep_files" "$expected")
    osgrep_recall=$(calculate_recall "$osgrep_files" "$expected")
    
    # Calculate tool overlap (do they agree?)
    overlap=$(calculate_overlap "$sgrep_csv" "$osgrep_csv")
    
    SGREP_PRECISIONS+=("$sgrep_precision")
    SGREP_RECALLS+=("$sgrep_recall")
    OSGREP_PRECISIONS+=("$osgrep_precision")
    OSGREP_RECALLS+=("$osgrep_recall")
    OVERLAPS+=("$overlap")
    
    echo "  sgrep:  precision=$sgrep_precision recall=$sgrep_recall"
    echo "  osgrep: precision=$osgrep_precision recall=$osgrep_recall"
    echo "  overlap: $overlap files in common"
    echo ""
    
    # Write detailed results
    cat >> "$RESULTS_FILE" << EOF
## Query: "$query"

**Expected files**: $expected

### sgrep results:
\`\`\`
$sgrep_files
\`\`\`
Precision: $sgrep_precision | Recall: $sgrep_recall

### osgrep results:
\`\`\`
$osgrep_files
\`\`\`
Precision: $osgrep_precision | Recall: $osgrep_recall

**Overlap**: $overlap files in common

---

EOF
done

# ========================================
# AGGREGATE STATISTICS
# ========================================

echo "=== Aggregate Statistics ===" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Calculate averages using Python
avg_sgrep_precision=$(python3 -c "
precs = [${SGREP_PRECISIONS[*]}]
print(f'{sum(map(float, precs)) / len(precs):.3f}')
" 2>/dev/null || echo "N/A")

avg_sgrep_recall=$(python3 -c "
recs = [${SGREP_RECALLS[*]}]
print(f'{sum(map(float, recs)) / len(recs):.3f}')
" 2>/dev/null || echo "N/A")

avg_osgrep_precision=$(python3 -c "
precs = [${OSGREP_PRECISIONS[*]}]
print(f'{sum(map(float, precs)) / len(precs):.3f}')
" 2>/dev/null || echo "N/A")

avg_osgrep_recall=$(python3 -c "
recs = [${OSGREP_RECALLS[*]}]
print(f'{sum(map(float, recs)) / len(recs):.3f}')
" 2>/dev/null || echo "N/A")

avg_overlap=$(python3 -c "
overlaps = [${OVERLAPS[*]}]
print(f'{sum(map(float, overlaps)) / len(overlaps):.1f}')
" 2>/dev/null || echo "N/A")

echo "| Metric | sgrep | osgrep |" | tee -a "$RESULTS_FILE"
echo "|--------|-------|--------|" | tee -a "$RESULTS_FILE"
echo "| Avg Precision | $avg_sgrep_precision | $avg_osgrep_precision |" | tee -a "$RESULTS_FILE"
echo "| Avg Recall | $avg_sgrep_recall | $avg_osgrep_recall |" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Average result overlap: $avg_overlap files" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# ========================================
# MODEL COMPARISON NOTE
# ========================================

cat << 'EOF' | tee -a "$RESULTS_FILE"
## Technical Notes

**sgrep**: 
- Embedding: nomic-embed-text-v1.5 (768 dimensions)
- Backend: llama.cpp (native C++)
- Search: In-memory L2 distance

**osgrep**:
- Embedding: mxbai-embed-xsmall-v1 (384 dimensions)
- Backend: transformers.js (WASM)
- Search: LanceDB

**Interpretation**:
- Higher precision = fewer irrelevant results returned
- Higher recall = more expected results found
- Overlap = agreement between tools (different != wrong)

EOF

echo ""
echo "Detailed results saved to: $RESULTS_FILE"
