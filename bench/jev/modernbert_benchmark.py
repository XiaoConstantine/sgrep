#!/usr/bin/env python3
"""Benchmark pinned GTE ModernBERT through sgrep's existing rerank boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
REVISION = "f7481e6055501a30fb19d090657df9ec1f79ab2c"


def load_shared():
    path = Path(__file__).with_name("coderank_benchmark.py")
    spec = importlib.util.spec_from_file_location("coderank_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nomic_query(query: str) -> np.ndarray:
    body = json.dumps({"content": "search_query: " + query}).encode()
    request = urllib.request.Request(
        "http://localhost:8080/embedding",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    vector = np.asarray(payload[0]["embedding"][0], dtype=np.float32)
    return vector / max(np.linalg.norm(vector), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--dataset", default="bench/jev/dataset.json")
    parser.add_argument("--output", default="bench/jev/results-modernbert.json")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    shared = load_shared()
    connection = sqlite3.connect(args.database)
    rows = connection.execute(
        "SELECT id, filepath, content, start_line, end_line, metadata, is_test "
        "FROM documents ORDER BY rowid"
    ).fetchall()
    docs = {}
    for doc_id, path, content, start, end, metadata, is_test in rows:
        description = (json.loads(metadata or "{}").get("description") or "").strip()
        text = (description + "\n\n" + content.strip()).strip() if description else content.strip()
        docs[doc_id] = {
            "path": path,
            "start": start,
            "end": end,
            "is_test": bool(is_test),
            "text": text,
        }

    vector_ids, vector_matrix = shared.load_sgrep_vectors(
        Path(args.database).parent / "vectors.mmap"
    )
    dataset = json.loads(Path(args.dataset).read_text())
    cases = [case for case in dataset["cases"] if case["split"] == "heldout"]
    lexical = {}
    for case in cases:
        terms = shared.hybrid_terms(case["query"])
        lexical[case["query"]] = [
            row[0]
            for row in connection.execute(
                "SELECT d.id FROM documents_fts f "
                "JOIN documents d ON d.rowid=f.rowid "
                "WHERE documents_fts MATCH ? "
                "ORDER BY bm25(documents_fts) ASC LIMIT 250",
                (terms,),
            )
        ] if terms else []

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    records = []
    latencies = []
    ranking_fingerprints = []
    score_ranges = []
    for run in range(args.runs):
        run_fingerprint = []
        for case in cases:
            started = time.perf_counter()
            query_vector = nomic_query(case["query"])
            similarities = vector_matrix @ query_vector
            dense_indices = np.argpartition(-similarities, 249)[:250]
            dense_indices = dense_indices[np.argsort(-similarities[dense_indices])]
            dense = [
                vector_ids[index]
                for index in dense_indices
                if 1 - similarities[index] <= 1.5
            ]
            balanced = shared.rrf(dense, lexical[case["query"]], 50)
            top = balanced[:20]
            pairs = [(case["query"], docs[doc_id]["text"]) for doc_id in top]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            with torch.inference_mode():
                scores = model(**inputs, return_dict=True).logits.view(-1).float().tolist()
            reranked = shared.fuse_reranker_scores(
                balanced, docs, case["query"], scores
            )
            final = shared.final_results(reranked, docs)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            run_fingerprint.append(final)
            if run == 0:
                score_ranges.append({
                    "query": case["query"],
                    "minimum": min(scores),
                    "maximum": max(scores),
                })
                records.append({
                    "query": case["query"],
                    "answers_query": case["answers_query"],
                    "metrics": shared.metrics(final, docs, case),
                    "latency_ms": elapsed,
                    "results": [
                        {
                            "file": docs[doc_id]["path"],
                            "start": docs[doc_id]["start"],
                            "end": docs[doc_id]["end"],
                        }
                        for doc_id in final
                    ],
                })
        ranking_fingerprints.append(run_fingerprint)

    answered = [record for record in records if record["answers_query"]]
    artifact = {
        "experiment": "isolated GTE ModernBERT reranker",
        "model": MODEL,
        "model_revision": REVISION,
        "license": "Apache-2.0",
        "corpus_commit": dataset["corpus_hash"],
        "runs": args.runs,
        "model_load_seconds": load_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mrr": sum(record["metrics"]["mrr"] for record in answered) / len(answered),
        "ndcg@10": sum(record["metrics"]["ndcg@10"] for record in answered) / len(answered),
        "recall@10": sum(record["metrics"]["recall@10"] for record in answered) / len(answered),
        "promotion_errors": sum(record["metrics"]["promotion_errors"] for record in answered),
        "p50_latency_ms": shared.percentile(latencies, 0.5),
        "p95_latency_ms": shared.percentile(latencies, 0.95),
        "rankings_identical_across_runs": all(
            fingerprint == ranking_fingerprints[0]
            for fingerprint in ranking_fingerprints[1:]
        ),
        "answerability": "Unsupported: no calibrated threshold; both no-answer controls return results.",
        "fallback": "The experiment wrapper returns the unchanged balanced order on model failure; production was not modified.",
        "privacy": "Local CPU inference only; no query or source content left the orb.",
        "score_ranges": score_ranges,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        f'MRR={artifact["mrr"]:.3f}',
        f'NDCG={artifact["ndcg@10"]:.3f}',
        f'R10={artifact["recall@10"]:.3f}',
        f'errors={artifact["promotion_errors"]}',
        f'p50/p95={artifact["p50_latency_ms"]:.1f}/{artifact["p95_latency_ms"]:.1f}ms',
    )


if __name__ == "__main__":
    main()
