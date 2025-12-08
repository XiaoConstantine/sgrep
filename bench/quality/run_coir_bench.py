#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "tqdm", "numpy"]
# ///
"""
Benchmark sgrep against CoIR (Code Information Retrieval) datasets.

CoIR is a comprehensive benchmark from ACL 2025 covering code retrieval tasks.
This script evaluates sgrep on standard code search benchmarks to enable
comparison with other embedding models and retrieval systems.

Usage:
  uv run bench/quality/run_coir_bench.py                    # Run on cosqa (default, fast mode)
  uv run bench/quality/run_coir_bench.py --dataset cosqa
  uv run bench/quality/run_coir_bench.py --full             # Full corpus (official CoIR eval)
  uv run bench/quality/run_coir_bench.py --full --workers 8 # Full corpus with 8 parallel workers
  uv run bench/quality/run_coir_bench.py --all              # Run all datasets
  uv run bench/quality/run_coir_bench.py --limit 100        # Limit queries for quick test

Modes:
  Default (fast): Only writes relevant docs to corpus - good for quick iteration
  --full: Writes entire corpus (20K+ docs) - matches official CoIR leaderboard eval
  --workers N: Number of parallel query workers (default: 6)

Metrics reported:
  - NDCG@10 (primary metric, same as CoIR leaderboard)
  - MRR (Mean Reciprocal Rank)
  - Recall@10
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Supported CoIR datasets with their HuggingFace paths and code column names
# Using official CoIR-Retrieval datasets which are maintained and work with latest datasets library
DATASETS = {
    "cosqa": {
        "path": "CoIR-Retrieval/cosqa",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "split": "corpus",
        "queries_split": "queries",
        "qrels_split": "default",
        "language": "python",
        "coir_format": True,  # Uses CoIR's standard format with separate corpus/queries/qrels
    },
    "codesearchnet-go": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "go-corpus",
        "queries_config": "go-queries",
        "qrels_config": "go-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "go",
        "coir_format": True,
        "per_lang": True,
    },
    "codesearchnet-python": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "python-corpus",
        "queries_config": "python-queries",
        "qrels_config": "python-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "python",
        "coir_format": True,
        "per_lang": True,
    },
    "codesearchnet-java": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "java-corpus",
        "queries_config": "java-queries",
        "qrels_config": "java-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "java",
        "coir_format": True,
        "per_lang": True,
    },
    "codesearchnet-javascript": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "javascript-corpus",
        "queries_config": "javascript-queries",
        "qrels_config": "javascript-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "javascript",
        "coir_format": True,
        "per_lang": True,
    },
    "codesearchnet-ruby": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "ruby-corpus",
        "queries_config": "ruby-queries",
        "qrels_config": "ruby-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "ruby",
        "coir_format": True,
        "per_lang": True,
    },
    "codesearchnet-php": {
        "path": "CoIR-Retrieval/codesearchnet",
        "corpus_config": "php-corpus",
        "queries_config": "php-queries",
        "qrels_config": "php-qrels",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "language": "php",
        "coir_format": True,
        "per_lang": True,
    },
    "apps": {
        "path": "CoIR-Retrieval/apps",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "split": "corpus",
        "queries_split": "queries",
        "qrels_split": "default",
        "language": "python",
        "coir_format": True,
    },
    "stackoverflow-qa": {
        "path": "CoIR-Retrieval/stackoverflow-qa",
        "query_col": "query",
        "doc_col": "text",
        "doc_id_col": "_id",
        "split": "corpus",
        "queries_split": "queries",
        "qrels_split": "default",
        "language": "mixed",
        "coir_format": True,
    },
}

FILE_EXTENSIONS = {
    "python": ".py",
    "go": ".go",
    "java": ".java",
    "javascript": ".js",
    "ruby": ".rb",
    "php": ".php",
}


def compute_ndcg(relevances: list[int], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        relevances: List of relevance scores (1 = relevant, 0 = not relevant)
                   in the order returned by the system
        k: Cutoff position

    Returns:
        NDCG@k score between 0 and 1
    """
    relevances = relevances[:k]

    # DCG: sum of rel_i / log2(i+2) for i in 0..k-1
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    # Ideal DCG: best possible ordering (all 1s first)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_mrr(relevances: list[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(relevances: list[int], total_relevant: int, k: int = 10) -> float:
    """Compute Recall@k."""
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r > 0)
    return hits / total_relevant


class CoIRBenchmark:
    """Benchmark runner for CoIR datasets."""

    def __init__(self, sgrep_bin: str = "sgrep", corpus_dir: str | None = None, reuse_index: bool = False):
        self.sgrep_bin = sgrep_bin
        self.temp_dir: Path | None = None
        self.corpus_dir: Path | None = None
        self.persistent_corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.reuse_index = reuse_index
        self._owns_temp_dir = False  # Track if we created the temp dir

    def prepare_corpus(self, dataset_name: str, limit: int | None = None, full_corpus: bool = False, sample_size: int | None = None) -> tuple[dict[str, str], list[dict]]:
        """
        Download dataset and prepare corpus files for sgrep indexing.

        Args:
            dataset_name: Name of the dataset to load
            limit: Max number of queries to evaluate
            full_corpus: If True, write all docs (official eval). If False, only relevant docs (fast mode).
            sample_size: If set, randomly sample this many docs from corpus (plus relevant docs).

        Returns:
            (doc_id_to_filename mapping, list of query dicts with ground truth)
        """
        config = DATASETS[dataset_name]
        print(f"Loading dataset: {config['path']}...")

        # CoIR format: separate corpus, queries, and qrels splits
        if config.get("coir_format"):
            return self._prepare_coir_corpus(config, limit, full_corpus, sample_size)

        # Legacy format (single dataset with all fields)
        return self._prepare_legacy_corpus(config, limit)

    def _prepare_coir_corpus(self, config: dict, limit: int | None = None, full_corpus: bool = False, sample_size: int | None = None) -> tuple[dict[str, str], list[dict]]:
        """Handle CoIR standard format with separate corpus/queries/qrels."""

        # Check if this is a per-language config (like codesearchnet)
        if config.get("per_lang"):
            corpus_config = config["corpus_config"]
            queries_config = config["queries_config"]
            qrels_config = config["qrels_config"]
        else:
            corpus_config = "corpus"
            queries_config = "queries"
            qrels_config = "default"

        # Load corpus documents
        print(f"Loading corpus from {config['path']} ({corpus_config})...")
        corpus = load_dataset(config["path"], corpus_config, split="corpus")

        # Load queries
        print(f"Loading queries ({queries_config})...")
        queries_ds = load_dataset(config["path"], queries_config, split="queries")

        # Load relevance judgments (qrels) - use test split
        print(f"Loading qrels ({qrels_config})...")
        qrels = load_dataset(config["path"], qrels_config, split="test")

        print(f"Corpus: {len(corpus)} docs, Queries: {len(queries_ds)}, Qrels: {len(qrels)}")

        # Use persistent directory if specified, otherwise create temp
        if self.persistent_corpus_dir:
            self.corpus_dir = self.persistent_corpus_dir / config['path'].split('/')[-1]
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            self._owns_temp_dir = False
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"coir_{config['path'].split('/')[-1]}_"))
            self.corpus_dir = self.temp_dir / "corpus"
            self.corpus_dir.mkdir()
            self._owns_temp_dir = True

        # Check if we can reuse existing index
        index_exists = (self.corpus_dir / ".sgrep").exists()
        skip_corpus_write = self.reuse_index and index_exists
        if skip_corpus_write:
            print(f"Reusing existing index at {self.corpus_dir}")

        # Build qrels lookup: query_id -> {doc_id: relevance}
        qrels_dict: dict[str, dict[str, int]] = {}
        for item in qrels:
            qid = str(item.get("query-id", item.get("query_id", "")))
            did = str(item.get("corpus-id", item.get("corpus_id", "")))
            score = int(item.get("score", 1))
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][did] = score

        print(f"Qrels entries: {len(qrels_dict)} unique queries with relevance judgments")

        # Build query lookup by ID
        query_lookup = {str(q.get("_id", "")): q for q in queries_ds}

        # Only use queries that have qrels (ground truth)
        valid_query_ids = [qid for qid in qrels_dict.keys() if qid in query_lookup]
        print(f"Queries with ground truth: {len(valid_query_ids)}")

        # Apply limit to queries that have qrels
        if limit:
            valid_query_ids = valid_query_ids[:limit]

        # Get relevant doc IDs for selected queries
        relevant_doc_ids = set()
        for qid in valid_query_ids:
            if qid in qrels_dict:
                relevant_doc_ids.update(qrels_dict[qid].keys())

        print(f"Relevant docs needed: {len(relevant_doc_ids)}")

        # Build corpus doc ID lookup
        ext = FILE_EXTENSIONS.get(config["language"], ".txt")
        doc_id_to_filename: dict[str, str] = {}

        # Skip writing if reusing existing index
        if skip_corpus_write:
            # Just build the doc_id_to_filename mapping from existing files
            for filepath in self.corpus_dir.glob(f"*{ext}"):
                doc_id = filepath.stem
                doc_id_to_filename[doc_id] = filepath.name
            print(f"Found {len(doc_id_to_filename)} existing corpus files")
        else:
            # Write corpus files
            if sample_size:
                print(f"Writing SAMPLED corpus ({sample_size} docs)...")
            elif full_corpus:
                print("Writing FULL corpus (official CoIR evaluation mode)...")
            else:
                print("Writing relevant docs only (fast mode - use --full for official eval)...")

            # If sampling, select random indices
            corpus_indices = list(range(len(corpus)))
            if sample_size and sample_size < len(corpus):
                import random
                random.seed(42)  # Reproducible sampling
                # Always include relevant docs, sample the rest
                sampled_indices = set(random.sample(corpus_indices, min(sample_size, len(corpus_indices))))
                # Add indices for relevant docs
                for i, doc in enumerate(corpus):
                    doc_id = str(doc.get("_id", f"doc_{i}"))
                    if doc_id in relevant_doc_ids:
                        sampled_indices.add(i)
                corpus_indices = sorted(sampled_indices)
                print(f"Sampled {len(corpus_indices)} docs (including {len(relevant_doc_ids)} relevant)")

            written = 0
            for i in tqdm(corpus_indices, desc="Processing corpus"):
                doc = corpus[i]
                doc_id = str(doc.get("_id", f"doc_{i}"))

                # In fast mode (no sampling), only write docs that are relevant to some query
                if not full_corpus and not sample_size and doc_id not in relevant_doc_ids:
                    continue

                doc_content = doc.get("text", "")
                if not doc_content:
                    continue

                # Sanitize filename
                safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in doc_id)[:80]
                filename = f"{safe_id}{ext}"
                filepath = self.corpus_dir / filename

                with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(doc_content)

                doc_id_to_filename[doc_id] = filename
                written += 1

            print(f"Created {written} corpus files")

        # Build queries with ground truth (using valid_query_ids from earlier)
        queries: list[dict] = []
        for qid in valid_query_ids:
            q = query_lookup.get(qid)
            if not q:
                continue

            query_text = q.get("text", "")

            if not query_text or len(query_text.strip()) < 5:
                continue

            # Get relevant docs from qrels
            relevant_docs = []
            if qid in qrels_dict:
                for did, score in qrels_dict[qid].items():
                    if did in doc_id_to_filename and score > 0:
                        relevant_docs.append(doc_id_to_filename[did])

            if relevant_docs:
                queries.append({
                    "query": query_text.strip(),
                    "relevant_docs": relevant_docs,
                    "query_id": qid,
                })

        print(f"Generated {len(queries)} queries with ground truth")
        return doc_id_to_filename, queries

    def _prepare_legacy_corpus(self, config: dict, limit: int | None = None) -> tuple[dict[str, str], list[dict]]:
        """Handle legacy format (single dataset with all fields)."""

        # Load from HuggingFace
        if "name" in config:
            ds = load_dataset(config["path"], config["name"], split=config["split"])
        else:
            ds = load_dataset(config["path"], split=config["split"])

        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        print(f"Loaded {len(ds)} examples")

        # Create temp directory for corpus
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"coir_legacy_"))
        self.corpus_dir = self.temp_dir / "corpus"
        self.corpus_dir.mkdir()

        ext = FILE_EXTENSIONS.get(config["language"], ".txt")
        doc_id_to_filename: dict[str, str] = {}
        queries: list[dict] = []

        print("Preparing corpus files...")
        for i, item in enumerate(tqdm(ds, desc="Writing files")):
            # Get document content and ID
            doc_content = item[config["doc_col"]]

            # Generate unique doc ID
            if config["doc_id_col"] in item and item[config["doc_id_col"]]:
                doc_id = str(item[config["doc_id_col"]])
            else:
                doc_id = f"doc_{i}"

            # Sanitize filename
            safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in doc_id)[:100]
            filename = f"{safe_id}_{i}{ext}"
            filepath = self.corpus_dir / filename

            # Write code to file
            with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                f.write(doc_content)

            doc_id_to_filename[doc_id] = filename

            # Build query with ground truth
            # In CodeSearchNet, each doc's docstring is its own query (self-retrieval task)
            query_text = item.get(config["query_col"], "")
            if query_text and len(query_text.strip()) > 10:
                queries.append({
                    "query": query_text.strip(),
                    "relevant_docs": [filename],  # Ground truth: the doc this query came from
                    "doc_id": doc_id,
                })

        print(f"Created {len(doc_id_to_filename)} corpus files in {self.corpus_dir}")
        print(f"Generated {len(queries)} queries with ground truth")

        return doc_id_to_filename, queries

    def index_corpus(self) -> bool:
        """Index the corpus with sgrep."""
        if not self.corpus_dir:
            return False

        # Check if we can reuse existing index
        index_path = self.corpus_dir / ".sgrep"
        if self.reuse_index and index_path.exists():
            print(f"\nReusing existing index at {index_path}")
            return True

        print(f"\nIndexing corpus with sgrep...")
        start = time.time()

        try:
            result = subprocess.run(
                [self.sgrep_bin, "index", "."],
                cwd=self.corpus_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for large corpora
            )

            elapsed = time.time() - start
            print(f"Indexing completed in {elapsed:.1f}s")

            if result.returncode != 0:
                print(f"Indexing failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print("Indexing timed out")
            return False
        except Exception as e:
            print(f"Indexing error: {e}")
            return False

    def run_query(self, query: str, k: int = 10, hybrid: bool = True, colbert: bool = True, rerank: bool = False) -> list[str]:
        """
        Run a single query and return ranked list of filenames.
        """
        if not self.corpus_dir:
            return []

        cmd = [self.sgrep_bin, "-n", str(k)]
        if hybrid:
            cmd.append("--hybrid")
        if colbert:
            cmd.append("--colbert")
        if rerank:
            cmd.extend(["--rerank", "--rerank-topk", "100"])
        cmd.append(query)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.corpus_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse results: format is path:start-end or just path
            filenames = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Extract filename from path:lines format
                path_part = line.split(":")[0] if ":" in line else line
                filename = os.path.basename(path_part)
                if filename and filename not in filenames:
                    filenames.append(filename)

            return filenames

        except Exception as e:
            print(f"Query error: {e}")
            return []

    def _run_single_query(self, args: tuple) -> dict:
        """Run a single query and return metrics. Used for parallel execution."""
        q, k, hybrid, colbert, rerank = args
        query_text = q["query"]
        relevant_docs = set(q["relevant_docs"])

        start = time.time()
        results = self.run_query(query_text, k=k, hybrid=hybrid, colbert=colbert, rerank=rerank)
        latency = (time.time() - start) * 1000

        # Build relevance vector
        relevances = [1 if r in relevant_docs else 0 for r in results]
        # Pad to k if needed
        relevances.extend([0] * (k - len(relevances)))

        ndcg = compute_ndcg(relevances, k)
        mrr = compute_mrr(relevances)
        recall = compute_recall_at_k(relevances, len(relevant_docs), k)

        return {
            "query": query_text,
            "relevant_docs": relevant_docs,
            "results": results,
            "ndcg": ndcg,
            "mrr": mrr,
            "recall": recall,
            "latency": latency,
        }

    def evaluate(
        self,
        queries: list[dict],
        k: int = 10,
        hybrid: bool = True,
        colbert: bool = True,
        rerank: bool = False,
        verbose: bool = False,
        num_workers: int = 6,
    ) -> dict[str, float]:
        """
        Evaluate sgrep on the query set using parallel execution.

        Returns dict with NDCG@10, MRR, Recall@10, and timing stats.
        """
        print(f"\nEvaluating {len(queries)} queries with {num_workers} workers...")

        # Prepare args for parallel execution
        query_args = [(q, k, hybrid, colbert, rerank) for q in queries]

        ndcg_scores = []
        mrr_scores = []
        recall_scores = []
        latencies = []

        # Run queries in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._run_single_query, args): i for i, args in enumerate(query_args)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Running queries"):
                result = future.result()

                ndcg_scores.append(result["ndcg"])
                mrr_scores.append(result["mrr"])
                recall_scores.append(result["recall"])
                latencies.append(result["latency"])

                if verbose and result["ndcg"] < 0.5:
                    print(f"\n  Low score query: '{result['query'][:60]}...'")
                    print(f"    Expected: {list(result['relevant_docs'])[:3]}")
                    print(f"    Got: {result['results'][:5]}")
                    print(f"    NDCG: {result['ndcg']:.3f}, MRR: {result['mrr']:.3f}")

        return {
            "ndcg@10": float(np.mean(ndcg_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "recall@10": float(np.mean(recall_scores)),
            "mean_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "num_queries": len(queries),
        }

    def cleanup(self):
        """Remove temporary files (only if we created them)."""
        if self._owns_temp_dir and self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up {self.temp_dir}")
        elif self.persistent_corpus_dir:
            print(f"Keeping persistent corpus at {self.corpus_dir}")


def run_benchmark(
    dataset_name: str,
    sgrep_bin: str = "sgrep",
    limit: int | None = None,
    hybrid: bool = True,
    colbert: bool = True,
    rerank: bool = False,
    verbose: bool = False,
    full_corpus: bool = False,
    num_workers: int = 6,
    corpus_dir: str | None = None,
    reuse_index: bool = False,
    sample_size: int | None = None,
    query_sample: int | None = None,
) -> dict[str, Any]:
    """Run full benchmark on a dataset."""

    print(f"\n{'='*60}")
    print(f"CoIR Benchmark: {dataset_name}")
    if sample_size:
        print(f"Mode: SAMPLED CORPUS ({sample_size} docs)")
    elif full_corpus:
        print("Mode: FULL CORPUS (official CoIR evaluation)")
    else:
        print("Mode: Fast (relevant docs only)")
    if reuse_index:
        print("Index reuse: ENABLED")
    print(f"{'='*60}")

    bench = CoIRBenchmark(sgrep_bin, corpus_dir=corpus_dir, reuse_index=reuse_index)

    try:
        # Prepare corpus
        _, queries = bench.prepare_corpus(dataset_name, limit=limit, full_corpus=full_corpus, sample_size=sample_size)

        if not queries:
            print("No queries generated!")
            return {}

        # Sample queries if requested
        if query_sample and query_sample < len(queries):
            import random
            random.seed(42)  # Reproducible
            queries = random.sample(queries, query_sample)
            print(f"Sampled {len(queries)} queries for evaluation")

        # Index (skip if --reuse-index since sgrep stores in ~/.sgrep/repos/)
        if reuse_index:
            print("Skipping indexing (--reuse-index set, assuming index exists in ~/.sgrep/repos/)")
        elif not bench.index_corpus():
            print("Failed to index corpus")
            return {}

        # Evaluate
        mode_label = []
        if hybrid:
            mode_label.append("hybrid")
        if colbert:
            mode_label.append("colbert")
        if rerank:
            mode_label.append("rerank")
        mode_str = "+".join(mode_label) if mode_label else "semantic"

        print(f"\nMode: {mode_str}")

        results = bench.evaluate(
            queries,
            k=10,
            hybrid=hybrid,
            colbert=colbert,
            rerank=rerank,
            verbose=verbose,
            num_workers=num_workers,
        )

        results["dataset"] = dataset_name
        results["mode"] = mode_str

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {dataset_name} ({mode_str})")
        print(f"{'='*60}")
        print(f"  NDCG@10:     {results['ndcg@10']:.4f}")
        print(f"  MRR:         {results['mrr']:.4f}")
        print(f"  Recall@10:   {results['recall@10']:.4f}")
        print(f"  Latency:     {results['mean_latency_ms']:.1f}ms (p50: {results['p50_latency_ms']:.1f}ms, p95: {results['p95_latency_ms']:.1f}ms)")
        print(f"  Queries:     {results['num_queries']}")

        return results

    finally:
        bench.cleanup()


def main():
    # Parse arguments
    dataset = "cosqa"
    limit: int | None = None
    run_all = False
    hybrid = True
    colbert = True
    rerank = False
    verbose = False
    sgrep_bin = "sgrep"
    full_corpus = False
    num_workers = 6
    corpus_dir: str | None = None
    reuse_index = False
    sample_size: int | None = None
    query_sample: int | None = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--query-sample" and i + 1 < len(args):
            query_sample = int(args[i + 1])
            i += 2
        elif arg == "--sample" and i + 1 < len(args):
            sample_size = int(args[i + 1])
            i += 2
        elif arg == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif arg == "--sgrep" and i + 1 < len(args):
            sgrep_bin = args[i + 1]
            i += 2
        elif arg == "--workers" and i + 1 < len(args):
            num_workers = int(args[i + 1])
            i += 2
        elif arg == "--corpus-dir" and i + 1 < len(args):
            corpus_dir = args[i + 1]
            i += 2
        elif arg == "--all":
            run_all = True
            i += 1
        elif arg == "--full":
            full_corpus = True
            i += 1
        elif arg == "--no-hybrid":
            hybrid = False
            i += 1
        elif arg == "--no-colbert":
            colbert = False
            i += 1
        elif arg == "--rerank":
            rerank = True
            i += 1
        elif arg == "--reuse-index":
            reuse_index = True
            i += 1
        elif arg == "--verbose" or arg == "-v":
            verbose = True
            i += 1
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            print("\nAvailable datasets:")
            for name in DATASETS:
                print(f"  - {name}")
            print("\nNew options:")
            print("  --corpus-dir DIR   Persistent directory for corpus (enables index reuse)")
            print("  --reuse-index      Skip indexing if .sgrep directory exists")
            print("  --sample N         Sample N docs from corpus (for large datasets)")
            print("  --query-sample N   Sample N queries for evaluation (faster testing)")
            sys.exit(0)
        else:
            i += 1

    all_results = []

    if run_all:
        datasets_to_run = list(DATASETS.keys())
    else:
        if dataset not in DATASETS:
            print(f"Unknown dataset: {dataset}")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run = [dataset]

    for ds_name in datasets_to_run:
        results = run_benchmark(
            ds_name,
            sgrep_bin=sgrep_bin,
            limit=limit,
            hybrid=hybrid,
            colbert=colbert,
            rerank=rerank,
            verbose=verbose,
            full_corpus=full_corpus,
            num_workers=num_workers,
            corpus_dir=corpus_dir,
            reuse_index=reuse_index,
            sample_size=sample_size,
            query_sample=query_sample,
        )
        if results:
            all_results.append(results)

    # Summary table if multiple datasets
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS DATASETS")
        print(f"{'='*80}")
        print(f"{'Dataset':<25} {'NDCG@10':>10} {'MRR':>10} {'Recall@10':>12} {'Latency':>10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['dataset']:<25} {r['ndcg@10']:>10.4f} {r['mrr']:>10.4f} {r['recall@10']:>12.4f} {r['mean_latency_ms']:>9.1f}ms")

        # Average
        avg_ndcg = np.mean([r["ndcg@10"] for r in all_results])
        avg_mrr = np.mean([r["mrr"] for r in all_results])
        avg_recall = np.mean([r["recall@10"] for r in all_results])
        avg_latency = np.mean([r["mean_latency_ms"] for r in all_results])
        print("-" * 80)
        print(f"{'AVERAGE':<25} {avg_ndcg:>10.4f} {avg_mrr:>10.4f} {avg_recall:>12.4f} {avg_latency:>9.1f}ms")

    # JSON output
    output_path = Path(__file__).parent / "coir_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
