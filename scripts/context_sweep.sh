#!/usr/bin/env bash
set -euo pipefail

# Reproducible 256/512-token indexing and quality/latency sweep.
# Usage: scripts/context_sweep.sh /path/to/corpus [sgrep-binary]
corpus=${1:?usage: context_sweep.sh /path/to/corpus [sgrep-binary]}
sgrep_bin=${2:-sgrep}
root=$(cd "$(dirname "$0")/.." && pwd)
results_dir="$root/bench/results/context-sweep"
base_port=${SGREP_SWEEP_BASE_PORT:-19080}
mkdir -p "$results_dir"

for context_tokens in 256 512; do
  port=$((base_port + context_tokens / 256 - 1))
  home="$results_dir/home-$context_tokens"
  rm -rf "$home"
  mkdir -p "$home/models"
  if [[ -d "${SGREP_HOME:-$HOME/.sgrep}/models" ]]; then
    rm -rf "$home/models"
    ln -s "${SGREP_HOME:-$HOME/.sgrep}/models" "$home/models"
  fi

  echo "== context=$context_tokens =="
  (
    cd "$corpus"
    /usr/bin/time -p env SGREP_HOME="$home" SGREP_PORT="$port" SGREP_CONTEXT_TOKENS="$context_tokens" \
      "$sgrep_bin" index .
  ) >"$results_dir/index-$context_tokens.log" 2>&1

  env SGREP_HOME="$home" SGREP_PORT="$port" SGREP_CONTEXT_TOKENS="$context_tokens" \
    uv run "$root/bench/quality/run_dspy_bench.py" \
      --tool sgrep --mode all --repo "$corpus" --sgrep "$sgrep_bin" \
      >"$results_dir/quality-$context_tokens.log" 2>&1
  env SGREP_HOME="$home" SGREP_PORT="$port" "$sgrep_bin" server stop >/dev/null 2>&1 || true
done

echo "Results: $results_dir"
