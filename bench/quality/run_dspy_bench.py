#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
"""
Benchmark semantic code search tools against dspy-go corpus.
Usage:
  uv run run_dspy_bench.py --tool sgrep --mode all   # Test all configurations
  uv run run_dspy_bench.py --tool sgrep              # Hybrid only (default)
  uv run run_dspy_bench.py --tool sgrep --rerank     # Hybrid + rerank
  uv run run_dspy_bench.py --tool sgrep --no-hybrid  # Semantic only
  uv run run_dspy_bench.py --tool sgrep --no-hybrid --rerank  # Semantic + rerank
  uv run run_dspy_bench.py --tool osgrep
  uv run run_dspy_bench.py --tool both
"""

import json
import subprocess
import sys
import time
import os
import re
from pathlib import Path

CORPUS = "/Users/xiao/development/github.com/XiaoConstantine/dspy-go"
DATASET_PATH = Path(__file__).parent / "dspy-go-dataset.json"
TOPK = 10


def run_sgrep(query: str, sgrep_bin: str, rerank: bool = False, hybrid: bool = True, colbert: bool = False) -> tuple[list[str], float]:
    """Run sgrep and return (result_files, latency_ms)."""
    cmd = [sgrep_bin, "-n", str(TOPK), "-q"]
    if hybrid:
        cmd.append("--hybrid")
    if rerank:
        cmd.extend(["--rerank", "--rerank-topk", "100"])
    if colbert:
        cmd.append("--colbert")
    cmd.append(query)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=CORPUS,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        output = ""
    except Exception as e:
        print(f"  Error: {e}")
        output = ""

    latency_ms = (time.time() - start) * 1000

    # Parse output: each line is "path/to/file.go:start-end" or similar
    files = []
    for line in output.strip().split('\n'):
        if not line:
            continue
        # Extract filename from path:lines format
        path_part = line.split(':')[0] if ':' in line else line
        filename = os.path.basename(path_part)
        if filename and filename not in files:  # Dedupe
            files.append(filename)

    return files, latency_ms


def run_osgrep(query: str) -> tuple[list[str], float]:
    """Run osgrep and return (result_files, latency_ms)."""
    cmd = ["osgrep", query, "-m", str(TOPK), "--compact"]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=CORPUS,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout + result.stderr  # osgrep may output to either
    except subprocess.TimeoutExpired:
        output = ""
    except Exception as e:
        print(f"  Error: {e}")
        output = ""

    latency_ms = (time.time() - start) * 1000

    # Parse osgrep output - it outputs file paths, one per line
    # Format may vary: could be "path/to/file.go" or "path/to/file.go:10-20"
    files = []
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('Searching') or line.startswith('Indexing'):
            continue
        # Remove ANSI color codes
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        # Extract path (before any colon for line numbers)
        path_part = line.split(':')[0] if ':' in line else line
        # Skip non-file lines
        if not path_part or path_part.startswith(' ') or '/' not in path_part and '.' not in path_part:
            continue
        filename = os.path.basename(path_part)
        if filename and filename.endswith('.go') and filename not in files:
            files.append(filename)

    return files, latency_ms


def compute_mrr(results: list[str], expected: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, r in enumerate(results):
        if r in expected:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(results: list[str], expected: list[str], k: int) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    hits = sum(1 for r in results[:k] if r in expected)
    return hits / k


def compute_recall_at_k(results: list[str], expected: list[str], k: int) -> float:
    """Compute Recall@K."""
    if len(expected) == 0:
        return 0.0
    hits = sum(1 for r in results[:k] if r in expected)
    return hits / len(expected)


def run_benchmark(tool: str, sgrep_bin: str = "sgrep", rerank: bool = False, hybrid: bool = True, colbert: bool = False):
    """Run benchmark for a specific tool."""
    # Load dataset
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    queries = dataset["queries"]

    tool_label = tool
    if tool == "sgrep":
        if hybrid and rerank and colbert:
            tool_label = "sgrep (cascade)"  # hybrid + colbert + rerank
        elif hybrid and rerank:
            tool_label = "sgrep (hybrid+rerank)"
        elif hybrid and colbert:
            tool_label = "sgrep (hybrid+colbert)"
        elif hybrid:
            tool_label = "sgrep (hybrid)"
        elif rerank:
            tool_label = "sgrep (semantic+rerank)"
        elif colbert:
            tool_label = "sgrep (semantic+colbert)"
        else:
            tool_label = "sgrep (semantic)"

    print(f"\n{'='*60}")
    print(f"Benchmarking: {tool_label}")
    print(f"Corpus: {CORPUS}")
    print(f"Top-K: {TOPK}")
    print(f"{'='*60}\n")

    # Collect metrics
    mrr_vals = []
    p5_vals = []
    p10_vals = []
    r5_vals = []
    r10_vals = []
    latency_vals = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        expected_files = [j["file"] for j in q.get("judgments", [])]

        if not expected_files:
            continue  # Skip queries without ground truth

        print(f"Query {i+1}/{len(queries)}: \"{query_text}\"")
        print(f"  Expected: {expected_files[:3]}{'...' if len(expected_files) > 3 else ''}")

        if tool == "sgrep":
            results, latency = run_sgrep(query_text, sgrep_bin, rerank, hybrid, colbert)
        else:  # osgrep
            results, latency = run_osgrep(query_text)

        mrr = compute_mrr(results, expected_files)
        p5 = compute_precision_at_k(results, expected_files, 5)
        p10 = compute_precision_at_k(results, expected_files, 10)
        r5 = compute_recall_at_k(results, expected_files, 5)
        r10 = compute_recall_at_k(results, expected_files, 10)

        mrr_vals.append(mrr)
        p5_vals.append(p5)
        p10_vals.append(p10)
        r5_vals.append(r5)
        r10_vals.append(r10)
        latency_vals.append(latency)

        print(f"  Results: {results[:5]}{'...' if len(results) > 5 else ''}")
        print(f"  MRR: {mrr:.3f} | P@5: {p5:.3f} | R@5: {r5:.3f} | Latency: {latency:.0f}ms")
        print()

    # Summary
    n = len(mrr_vals)
    if n == 0:
        print("No queries with ground truth!")
        return None

    summary = {
        "tool": tool_label,
        "queries": n,
        "mean_mrr": sum(mrr_vals)/n,
        "mean_p5": sum(p5_vals)/n,
        "mean_p10": sum(p10_vals)/n,
        "mean_r5": sum(r5_vals)/n,
        "mean_r10": sum(r10_vals)/n,
        "mean_latency_ms": sum(latency_vals)/n,
    }

    print(f"{'='*60}")
    print(f"SUMMARY: {tool_label}")
    print(f"{'='*60}")
    print(f"Mean MRR:      {summary['mean_mrr']:.3f}")
    print(f"Mean P@5:      {summary['mean_p5']:.3f}")
    print(f"Mean P@10:     {summary['mean_p10']:.3f}")
    print(f"Mean R@5:      {summary['mean_r5']:.3f}")
    print(f"Mean R@10:     {summary['mean_r10']:.3f}")
    print(f"Mean Latency:  {summary['mean_latency_ms']:.1f}ms")
    print(f"Queries: {n}")

    return summary


def main():
    tool = "sgrep"
    sgrep_bin = "sgrep"
    mode = "all"  # default: test all sgrep modes

    # Parse args
    rerank = "--rerank" in sys.argv
    no_hybrid = "--no-hybrid" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--tool" and i + 1 < len(sys.argv):
            tool = sys.argv[i + 1]
        if arg == "--sgrep" and i + 1 < len(sys.argv):
            sgrep_bin = sys.argv[i + 1]
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]

    summaries = []

    if tool in ["sgrep", "both"]:
        if mode == "all":
            # Test all configurations
            # 1. Semantic only
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=False, colbert=False)
            if s:
                summaries.append(s)
            # 2. Hybrid only
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=True, colbert=False)
            if s:
                summaries.append(s)
            # 3. Hybrid + ColBERT (no cross-encoder)
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=True, colbert=True)
            if s:
                summaries.append(s)
            # 4. CASCADE: Hybrid + ColBERT + Cross-encoder
            s = run_benchmark("sgrep", sgrep_bin, rerank=True, hybrid=True, colbert=True)
            if s:
                summaries.append(s)
        else:
            # Single configuration based on flags
            use_hybrid = not no_hybrid
            use_colbert = "--colbert" in sys.argv
            s = run_benchmark("sgrep", sgrep_bin, rerank=rerank, hybrid=use_hybrid, colbert=use_colbert)
            if s:
                summaries.append(s)

    if tool in ["osgrep", "both"]:
        s = run_benchmark("osgrep")
        if s:
            summaries.append(s)

    # Comparison table
    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Tool':<20} {'MRR':>8} {'P@5':>8} {'R@5':>8} {'Latency':>10}")
        print("-" * 60)
        for s in summaries:
            print(f"{s['tool']:<20} {s['mean_mrr']:>8.3f} {s['mean_p5']:>8.3f} {s['mean_r5']:>8.3f} {s['mean_latency_ms']:>9.0f}ms")

    # JSON output
    print("\n\nJSON Summaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
