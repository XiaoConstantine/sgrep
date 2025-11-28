#!/usr/bin/env bash
# Performance benchmark script with regression detection
#
# Usage:
#   ./scripts/perf_bench.sh              # Run benchmarks and compare to baseline
#   ./scripts/perf_bench.sh --save       # Run and save as new baseline
#   ./scripts/perf_bench.sh --profile    # Run with CPU profiling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/bench"
RESULTS_DIR="$BENCH_DIR/results"

mkdir -p "$RESULTS_DIR"

BASELINE_FILE="$RESULTS_DIR/bench_baseline.txt"
CURRENT_FILE="$RESULTS_DIR/bench_current.txt"
PROFILE_DIR="$RESULTS_DIR/profiles"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [--save|--profile|--quick|--help]"
    echo ""
    echo "Options:"
    echo "  --save      Save current results as new baseline"
    echo "  --profile   Run with CPU/memory profiling"
    echo "  --quick     Run only quick benchmarks (skip large tests)"
    echo "  --help      Show this help"
}

run_benchmarks() {
    local output_file="$1"
    local short_flag="${2:-}"

    echo -e "${GREEN}Running benchmarks...${NC}"
    
    if [ -n "$short_flag" ]; then
        go test ./... -run=^$ -bench=. -benchmem -tags=sqlite_vec -short \
            2>&1 | tee "$output_file"
    else
        go test ./... -run=^$ -bench=. -benchmem -tags=sqlite_vec \
            2>&1 | tee "$output_file"
    fi
}

run_with_profile() {
    mkdir -p "$PROFILE_DIR"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo -e "${GREEN}Running benchmarks with profiling...${NC}"
    
    # CPU profile
    go test ./internal/bench -run=^$ -bench=BenchmarkSearchEndToEnd -benchmem \
        -tags=sqlite_vec -cpuprofile="$PROFILE_DIR/cpu_$timestamp.prof" \
        -memprofile="$PROFILE_DIR/mem_$timestamp.prof"
    
    echo ""
    echo -e "${GREEN}Profiles saved:${NC}"
    echo "  CPU: $PROFILE_DIR/cpu_$timestamp.prof"
    echo "  Mem: $PROFILE_DIR/mem_$timestamp.prof"
    echo ""
    echo "View with:"
    echo "  go tool pprof -http=:8080 $PROFILE_DIR/cpu_$timestamp.prof"
}

compare_results() {
    if [ ! -f "$BASELINE_FILE" ]; then
        echo -e "${YELLOW}No baseline found. Run with --save to create one.${NC}"
        return
    fi

    echo ""
    echo -e "${GREEN}=== Benchmark Comparison ===${NC}"
    
    if command -v benchstat &> /dev/null; then
        benchstat "$BASELINE_FILE" "$CURRENT_FILE"
    else
        echo -e "${YELLOW}benchstat not installed. Install with:${NC}"
        echo "  go install golang.org/x/perf/cmd/benchstat@latest"
        echo ""
        echo "Showing raw diff:"
        diff "$BASELINE_FILE" "$CURRENT_FILE" || true
    fi
}

main() {
    cd "$PROJECT_ROOT"

    case "${1:-}" in
        --help|-h)
            usage
            exit 0
            ;;
        --save)
            run_benchmarks "$BASELINE_FILE"
            echo -e "${GREEN}Baseline saved to: $BASELINE_FILE${NC}"
            ;;
        --profile)
            run_with_profile
            ;;
        --quick)
            run_benchmarks "$CURRENT_FILE" "short"
            compare_results
            ;;
        "")
            run_benchmarks "$CURRENT_FILE"
            compare_results
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
