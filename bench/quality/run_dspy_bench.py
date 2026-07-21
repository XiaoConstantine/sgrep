#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken", "tqdm"]
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
  uv run run_dspy_bench.py --tool mgrep              # Mixedbread mgrep (cloud)
  uv run run_dspy_bench.py --tool all                # Test all tools
"""

import hashlib
import json
import subprocess
import sys
import time
import os
import re
import shutil
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import tiktoken
from tqdm import tqdm

# Initialize tokenizer (cl100k_base is used by GPT-4, Claude uses similar)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Cost per 1M tokens (Claude 3.5 Sonnet input pricing)
COST_PER_1M_TOKENS = 3.00


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(TOKENIZER.encode(text))


def estimate_cost(tokens: int) -> float:
    """Estimate cost in USD based on token count."""
    return (tokens / 1_000_000) * COST_PER_1M_TOKENS

CORPUS = "/Users/xiao/development/github.com/XiaoConstantine/dspy-go"
DATASET_PATH = Path(__file__).parent / "dspy-go-dataset.json"
TOPK = 10
CONCURRENCY = 1
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4
EVAL_SPLIT = "all"
EVAL_FOLD = 0
EVAL_FOLDS = 4


@lru_cache(maxsize=1024)
def read_file_range(filepath: str, start_line: int, end_line: int) -> str:
    """Read specific line range from a file. Cached to avoid repeated disk I/O."""
    try:
        with open(filepath, 'r', errors='ignore') as f:
            lines = f.readlines()
            # Convert to 0-indexed
            start = max(0, start_line - 1)
            end = min(len(lines), end_line)
            return ''.join(lines[start:end])
    except:
        return ""


def run_sgrep(query: str, sgrep_bin: str, rerank: bool = False, hybrid: bool = True, colbert: bool = False) -> tuple[list[str], float, float, int, dict]:
    """Run sgrep and return (result_files, search_latency_ms, total_latency_ms, tokens, timing_breakdown).

    Returns:
        result_files: List of result filenames
        search_latency_ms: Actual search time (from sgrep debug output)
        total_latency_ms: Total subprocess time (includes startup overhead)
        tokens: Token count of results
        timing_breakdown: Dict with pipeline stage timings
    """
    # Force a profile so local artifacts cannot silently contaminate modes.
    profile = "quality" if colbert else ("balanced" if hybrid else "fast")
    cmd = [
        sgrep_bin, "-n", str(TOPK), "-d", "--profile", profile,
        "--semantic-weight", str(SEMANTIC_WEIGHT),
        "--bm25-weight", str(BM25_WEIGHT),
    ]
    if rerank:
        cmd.extend(["--rerank", "--rerank-topk", "100"])
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
        stderr = result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"sgrep exited {result.returncode} for query {query!r}: "
                f"{stderr.strip() or output.strip()}"
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"sgrep timed out for query {query!r}") from exc

    total_latency_ms = (time.time() - start) * 1000

    # Parse search latency from debug output (stderr)
    # Format: [DEBUG] Search completed: 10 results in 42ms
    search_latency_ms = total_latency_ms  # fallback to total
    timing_breakdown = {}

    debug_output = stderr if stderr else output
    for line in debug_output.split('\n'):
        # Parse actual search time
        if 'Search completed:' in line and 'in ' in line:
            match = re.search(r'in (\d+)ms', line)
            if match:
                search_latency_ms = float(match.group(1))

        # Parse pipeline stage timings
        # Format:   query_embedding:     13ms (1 ops, 13.135ms avg, 43%)
        if ':' in line and 'ms' in line and ('query_embedding' in line or 'vector_search' in line or 'reranking' in line):
            match = re.match(r'\s*(\w+):\s+(\d+)ms', line)
            if match:
                stage = match.group(1)
                time_ms = int(match.group(2))
                timing_breakdown[stage] = time_ms

    # Parse output and collect content for token counting
    # Format: path/to/file.go:start-end
    files = []
    total_content = ""
    for line in output.strip().split('\n'):
        if not line or line.startswith('[DEBUG]'):
            continue
        # Parse path:lines format
        if ':' in line:
            path_part, line_range = line.rsplit(':', 1)
            if '-' in line_range:
                try:
                    start_line, end_line = map(int, line_range.split('-'))
                    full_path = os.path.join(CORPUS, path_part)
                    content = read_file_range(full_path, start_line, end_line)
                    total_content += f"// {path_part}:{line_range}\n{content}\n"
                except ValueError:
                    pass
        else:
            path_part = line

        relative_path = os.path.normpath(path_part).removeprefix("./")
        if relative_path and relative_path not in files:
            files.append(relative_path)

    tokens = count_tokens(total_content) if total_content else 0

    return files, search_latency_ms, total_latency_ms, tokens, timing_breakdown


def warmup_sgrep(sgrep_bin: str, corpus: str):
    """Run a warmup query to initialize llama-server connection."""
    cmd = [sgrep_bin, "warmup query", "--hybrid", "-n", "1"]
    try:
        result = subprocess.run(cmd, cwd=corpus, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("sgrep warmup timed out") from exc
    if result.returncode != 0:
        raise RuntimeError(f"sgrep warmup failed: {result.stderr.strip() or result.stdout.strip()}")


def run_osgrep(query: str) -> tuple[list[str], float, int]:
    """Run osgrep and return (result_files, latency_ms, tokens)."""
    # Run without --compact to get full content for token counting
    cmd = ["osgrep", query, "-m", str(TOPK)]

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
        if result.returncode != 0:
            raise RuntimeError(f"osgrep exited {result.returncode}: {output.strip()}")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"osgrep timed out for query {query!r}") from exc

    latency_ms = (time.time() - start) * 1000

    # Count tokens from full output
    tokens = count_tokens(output) if output else 0

    # Parse osgrep output - it outputs file paths, one per line
    # Format may vary: could be "path/to/file.go" or "path/to/file.go:10-20"
    files = []
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('Searching') or line.startswith('Indexing') or line.startswith('Worker'):
            continue
        # Remove ANSI color codes
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        # Extract path (before any colon for line numbers)
        path_part = line.split(':')[0] if ':' in line else line
        # Skip non-file lines
        if not path_part or path_part.startswith(' ') or '/' not in path_part and '.' not in path_part:
            continue
        relative_path = os.path.normpath(path_part).removeprefix("./")
        if relative_path.endswith('.go') and relative_path not in files:
            files.append(relative_path)

    return files, latency_ms, tokens


def run_mgrep(query: str, rerank: bool = True) -> tuple[list[str], float, int]:
    """Run mgrep (mixedbread cloud) and return (result_files, latency_ms, tokens)."""
    # Run with -c to get content for token counting
    cmd = ["mgrep", "search", query, "-m", str(TOPK), "-c"]
    if not rerank:
        cmd.append("--no-rerank")

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
        if result.returncode != 0:
            raise RuntimeError(
                f"mgrep exited {result.returncode} for query {query!r}: "
                f"{result.stderr.strip() or output.strip()}"
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"mgrep timed out for query {query!r}") from exc

    latency_ms = (time.time() - start) * 1000

    # Count tokens from full output
    tokens = count_tokens(output) if output else 0

    # Parse mgrep output: ./path/to/file.go:start-end (XX.XX% match)
    files = []
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove ANSI color codes
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        # Extract path (before colon for line numbers)
        # Format: ./path/to/file.go:123-456 (XX.XX% match)
        if ':' in line:
            path_part = line.split(':')[0]
        else:
            path_part = line.split(' ')[0] if ' ' in line else line
        # Remove leading ./
        if path_part.startswith('./'):
            path_part = path_part[2:]
        relative_path = os.path.normpath(path_part).removeprefix("./")
        if relative_path and relative_path not in files:
            files.append(relative_path)

    return files, latency_ms, tokens


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


@lru_cache(maxsize=512)
def resolve_corpus_path(judged_path: str) -> str:
    """Resolve legacy basename qrels to an unambiguous corpus-relative path."""
    normalized = os.path.normpath(judged_path).removeprefix("./")
    if "/" in normalized:
        return normalized
    matches = [p.relative_to(CORPUS).as_posix() for p in Path(CORPUS).rglob(normalized)]
    if len(matches) != 1:
        raise ValueError(f"qrel {judged_path!r} resolves to {len(matches)} files; use a full relative path")
    return matches[0]


def _run_single_query(args):
    """Helper function to run one query."""
    i, q, tool, sgrep_bin, rerank, hybrid, colbert = args
    query_text = q["query"]
    expected_files = [resolve_corpus_path(j["file"]) for j in q.get("judgments", [])]

    if not expected_files:
        return None  # Skip queries without ground truth

    if tool == "sgrep":
        results, search_latency, total_latency, tokens, timing = run_sgrep(query_text, sgrep_bin, rerank, hybrid, colbert)
    elif tool == "mgrep":
        results, total_latency, tokens = run_mgrep(query_text, rerank)
        search_latency = total_latency
        timing = {}
    else:  # osgrep
        results, total_latency, tokens = run_osgrep(query_text)
        search_latency = total_latency
        timing = {}

    mrr = compute_mrr(results, expected_files)
    p5 = compute_precision_at_k(results, expected_files, 5)
    p10 = compute_precision_at_k(results, expected_files, 10)
    r5 = compute_recall_at_k(results, expected_files, 5)
    r10 = compute_recall_at_k(results, expected_files, 10)

    return {
        "index": i,
        "query": query_text,
        "expected": expected_files,
        "results": results,
        "mrr": mrr,
        "p5": p5,
        "p10": p10,
        "r5": r5,
        "r10": r10,
        "search_latency": search_latency,
        "total_latency": total_latency,
        "timing_breakdown": timing,
        "tokens": tokens,
    }


def run_benchmark(tool: str, sgrep_bin: str = "sgrep", rerank: bool = False, hybrid: bool = True, colbert: bool = False):
    """Run benchmark for a specific tool."""
    # Load dataset
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    queries = dataset["queries"]
    if EVAL_SPLIT != "all":
        # Rank hashes before assigning folds so every fold has five examples on
        # the current 20-query qrel set instead of an accidental two-query test.
        ranked = sorted(
            queries,
            key=lambda query: hashlib.sha256(query["query"].encode()).hexdigest(),
        )
        test_queries = {query["query"] for rank, query in enumerate(ranked) if rank % EVAL_FOLDS == EVAL_FOLD}
        queries = [
            query for query in queries
            if (query["query"] in test_queries) == (EVAL_SPLIT == "test")
        ]

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

    # Warmup for sgrep to initialize llama-server connection
    if tool == "sgrep":
        print("Warming up sgrep (initializing llama-server connection)...")
        warmup_sgrep(sgrep_bin, CORPUS)

    # Collect metrics
    mrr_vals = []
    p5_vals = []
    p10_vals = []
    r5_vals = []
    r10_vals = []
    search_latency_vals = []
    total_latency_vals = []
    token_vals = []

    # Aggregate timing breakdowns
    timing_totals = {"query_embedding": [], "vector_search": [], "reranking": []}

    # Prepare args for parallel execution
    query_args = [
        (i, q, tool, sgrep_bin, rerank, hybrid, colbert)
        for i, q in enumerate(queries)
    ]

    # Sequential execution measures user-visible latency. Set --concurrency only
    # for a separate throughput/contention run.
    batch_started = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(tqdm(executor.map(_run_single_query, query_args), total=len(query_args), desc="Running queries"))
    batch_elapsed = time.time() - batch_started

    # Process results and print (sequential to maintain output order)
    for result in tqdm(results, desc="Processing results", disable=len([r for r in results if r is not None]) == 0):
        if result is None:
            continue

        i = result["index"]
        query_text = result["query"]
        expected_files = result["expected"]
        results_files = result["results"]
        mrr = result["mrr"]
        p5 = result["p5"]
        p10 = result["p10"]
        r5 = result["r5"]
        r10 = result["r10"]
        search_latency = result["search_latency"]
        total_latency = result["total_latency"]
        timing = result.get("timing_breakdown", {})
        tokens = result["tokens"]

        mrr_vals.append(mrr)
        p5_vals.append(p5)
        p10_vals.append(p10)
        r5_vals.append(r5)
        r10_vals.append(r10)
        search_latency_vals.append(search_latency)
        total_latency_vals.append(total_latency)
        token_vals.append(tokens)

        # Collect timing breakdown
        for stage in timing_totals:
            if stage in timing:
                timing_totals[stage].append(timing[stage])

        cost = estimate_cost(tokens)
        timing_str = ""
        if timing:
            timing_str = f" [emb:{timing.get('query_embedding', 0)}ms vec:{timing.get('vector_search', 0)}ms rerank:{timing.get('reranking', 0)}ms]"

        print(f"Query {i+1}/{len(queries)}: \"{query_text}\"")
        print(f"  Expected: {expected_files[:3]}{'...' if len(expected_files) > 3 else ''}")
        print(f"  Results: {results_files[:5]}{'...' if len(results_files) > 5 else ''}")
        print(f"  MRR: {mrr:.3f} | P@5: {p5:.3f} | R@5: {r5:.3f} | Search: {search_latency:.0f}ms | Total: {total_latency:.0f}ms{timing_str}")
        print()

    # Summary
    n = len(mrr_vals)
    if n == 0:
        print("No queries with ground truth!")
        return None

    total_tokens = sum(token_vals)
    total_cost = estimate_cost(total_tokens)

    # Calculate timing breakdown averages
    timing_avg = {}
    for stage, vals in timing_totals.items():
        if vals:
            timing_avg[stage] = sum(vals) / len(vals)

    summary = {
        "tool": tool_label,
        "queries": n,
        "mean_mrr": sum(mrr_vals)/n,
        "mean_p5": sum(p5_vals)/n,
        "mean_p10": sum(p10_vals)/n,
        "mean_r5": sum(r5_vals)/n,
        "mean_r10": sum(r10_vals)/n,
        "mean_search_latency_ms": sum(search_latency_vals)/n,
        "median_search_latency_ms": statistics.median(search_latency_vals),
        "p95_search_latency_ms": sorted(search_latency_vals)[max(0, int(0.95 * n + 0.999999) - 1)],
        "mean_total_latency_ms": sum(total_latency_vals)/n,
        "concurrency": CONCURRENCY,
        "throughput_queries_per_second": n / batch_elapsed if batch_elapsed > 0 else 0,
        "corpus_commit": subprocess.check_output(["git", "-C", CORPUS, "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_metadata": {
            "top_k": TOPK,
            "embedding_model": "nomic-embed-text-v1.5.Q8_0.gguf",
            "context_tokens": int(os.environ.get("SGREP_CONTEXT_TOKENS", "512")),
            "sgrep_binary_sha256": hashlib.sha256(Path(sgrep_bin).read_bytes()).hexdigest() if tool == "sgrep" else None,
            "profile": "quality" if colbert else ("balanced" if hybrid else "fast"),
            "rerank": rerank,
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "eval_split": EVAL_SPLIT,
            "eval_fold": EVAL_FOLD if EVAL_SPLIT != "all" else None,
            "eval_folds": EVAL_FOLDS,
        },
        "timing_breakdown": timing_avg,
        "total_tokens": total_tokens,
        "mean_tokens": total_tokens / n,
        "total_cost_usd": total_cost,
    }

    print(f"{'='*60}")
    print(f"SUMMARY: {tool_label}")
    print(f"{'='*60}")
    print(f"Mean MRR:          {summary['mean_mrr']:.3f}")
    print(f"Mean P@5:          {summary['mean_p5']:.3f}")
    print(f"Mean P@10:         {summary['mean_p10']:.3f}")
    print(f"Mean R@5:          {summary['mean_r5']:.3f}")
    print(f"Mean R@10:         {summary['mean_r10']:.3f}")
    print(f"Mean Search Time:  {summary['mean_search_latency_ms']:.1f}ms")
    print(f"Median / P95:      {summary['median_search_latency_ms']:.1f}ms / {summary['p95_search_latency_ms']:.1f}ms")
    print(f"Mean Total Time:   {summary['mean_total_latency_ms']:.1f}ms  (includes process startup)")
    if timing_avg:
        print(f"  Pipeline breakdown:")
        if 'query_embedding' in timing_avg:
            print(f"    - Query embedding: {timing_avg['query_embedding']:.1f}ms")
        if 'vector_search' in timing_avg:
            print(f"    - Vector search:   {timing_avg['vector_search']:.1f}ms")
        if 'reranking' in timing_avg:
            print(f"    - Reranking:       {timing_avg['reranking']:.1f}ms")
    print(f"Total Tokens:      {summary['total_tokens']:,}")
    print(f"Mean Tokens:       {summary['mean_tokens']:.0f}")
    print(f"Total Cost:        ${summary['total_cost_usd']:.4f}")
    print(f"Queries: {n}")

    return summary


def main():
    global CORPUS, CONCURRENCY, SEMANTIC_WEIGHT, BM25_WEIGHT, EVAL_SPLIT, EVAL_FOLD
    tool = "sgrep"
    sgrep_bin = "sgrep"
    mode = "all"  # default: test all sgrep modes

    # Parse args
    rerank = "--rerank" in sys.argv
    no_hybrid = "--no-hybrid" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--tool" and i + 1 < len(sys.argv):
            tool = sys.argv[i + 1]
        if arg in ["--sgrep", "--sgrep-path"] and i + 1 < len(sys.argv):
            sgrep_bin = sys.argv[i + 1]
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        if arg == "--repo" and i + 1 < len(sys.argv):
            CORPUS = sys.argv[i + 1]
        if arg == "--concurrency" and i + 1 < len(sys.argv):
            CONCURRENCY = max(1, int(sys.argv[i + 1]))
        if arg == "--semantic-weight" and i + 1 < len(sys.argv):
            SEMANTIC_WEIGHT = float(sys.argv[i + 1])
        if arg == "--bm25-weight" and i + 1 < len(sys.argv):
            BM25_WEIGHT = float(sys.argv[i + 1])
        if arg == "--split" and i + 1 < len(sys.argv):
            EVAL_SPLIT = sys.argv[i + 1]
            if EVAL_SPLIT not in {"all", "train", "test"}:
                raise SystemExit("--split must be all, train, or test")
        if arg == "--fold" and i + 1 < len(sys.argv):
            EVAL_FOLD = int(sys.argv[i + 1])
            if not 0 <= EVAL_FOLD < EVAL_FOLDS:
                raise SystemExit(f"--fold must be between 0 and {EVAL_FOLDS - 1}")

    with open(DATASET_PATH) as f:
        pinned = json.load(f).get("corpus_hash", "")
    actual_commit = subprocess.check_output(["git", "-C", CORPUS, "rev-parse", "HEAD"], text=True).strip()
    if pinned and pinned != actual_commit:
        raise SystemExit(f"corpus commit mismatch: dataset={pinned}, checkout={actual_commit}")

    # Resolve command names through PATH; only path-like values are relative to cwd.
    if not os.path.isabs(sgrep_bin):
        if os.sep not in sgrep_bin:
            resolved = shutil.which(sgrep_bin)
            if not resolved:
                raise SystemExit(f"sgrep binary not found in PATH: {sgrep_bin}")
            sgrep_bin = resolved
        else:
            sgrep_bin = os.path.abspath(sgrep_bin)

    summaries = []

    if tool in ["sgrep", "all"]:
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
        elif mode == "semantic":
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=False, colbert=False)
            if s:
                summaries.append(s)
        elif mode == "hybrid":
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=True, colbert=False)
            if s:
                summaries.append(s)
        elif mode == "hybrid+colbert":
            s = run_benchmark("sgrep", sgrep_bin, rerank=False, hybrid=True, colbert=True)
            if s:
                summaries.append(s)
        elif mode == "cascade":
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

    if tool in ["osgrep", "all"]:
        s = run_benchmark("osgrep")
        if s:
            summaries.append(s)

    if tool in ["mgrep", "all"]:
        # mgrep with rerank (default)
        s = run_benchmark("mgrep", rerank=True)
        if s:
            summaries.append(s)

    # Comparison table
    if len(summaries) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"{'Tool':<22} {'MRR':>7} {'P@5':>6} {'R@5':>6} {'Latency':>9} {'Tokens':>8} {'Cost':>8}")
        print("-" * 80)
        for s in summaries:
            print(f"{s['tool']:<22} {s['mean_mrr']:>7.3f} {s['mean_p5']:>6.3f} {s['mean_r5']:>6.3f} {s['mean_search_latency_ms']:>8.0f}ms {s['mean_tokens']:>8.0f} ${s['total_cost_usd']:>6.4f}")

    # JSON output
    print("\n\nJSON Summaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
