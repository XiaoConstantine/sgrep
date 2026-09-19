#!/usr/bin/env python3
"""Benchmark pinned LateOn-Code without changing sgrep production behavior."""

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
from pylate import models, rank


MODEL = "lightonai/LateOn-Code"
REVISION = "ace431824f35db231178fc602e33296784762a2e"


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


def rank_fuse(ids: list[str], lateon_order: list[str], docs: dict[str, dict], query: str) -> list[str]:
    """Apply the production 0.75/0.25 weights to ranks, not incomparable logits."""
    lateon_rank = {doc_id: position for position, doc_id in enumerate(lateon_order)}
    scored = []
    for position, doc_id in enumerate(ids):
        baseline_score = 1 / (61 + position)
        reranker_score = 1 / (61 + lateon_rank[doc_id]) if doc_id in lateon_rank else 0.0
        score = (0.75 * baseline_score + 0.25 * reranker_score) * docs[doc_id]["boost"]
        scored.append((doc_id, score))
    selected = {doc_id for doc_id, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:10]}
    # This matches SearchWithOptions: fused scores select candidates, then the
    # original-distance order is restored before filtering and deduplication.
    return [doc_id for doc_id in ids if doc_id in selected]


def result_id(item) -> str:
    if isinstance(item, dict):
        return str(item["id"])
    return str(item.id)


def result_score(item) -> float:
    if isinstance(item, dict):
        return float(item["score"])
    return float(item.score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--dataset", default="bench/jev/dataset.json")
    parser.add_argument("--output", default="bench/jev/results-lateon-code.json")
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
    model = models.ColBERT(
        model_name_or_path=MODEL,
        revision=REVISION,
        device="cpu",
    )
    load_seconds = time.perf_counter() - load_started

    records = []
    direct_records = []
    latencies = []
    ranking_fingerprints = []
    failures = []
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
            lateon_results = []
            lateon_order = balanced
            try:
                query_embeddings = model.encode(
                    [case["query"]], is_query=True, show_progress_bar=False
                )
                document_embeddings = model.encode(
                    [[docs[doc_id]["text"] for doc_id in top]],
                    is_query=False,
                    show_progress_bar=False,
                )
                lateon_results = rank.rerank(
                    documents_ids=[top],
                    queries_embeddings=query_embeddings,
                    documents_embeddings=document_embeddings,
                    device="cpu",
                )[0]
                lateon_order = [result_id(item) for item in lateon_results]
                for doc_id in balanced:
                    docs[doc_id]["boost"] = shared.code_boost(
                        docs[doc_id]["path"], shared.intent(case["query"])
                    )
                reranked = rank_fuse(balanced, lateon_order, docs, case["query"])
            except Exception as error:
                failures.append({"run": run, "query": case["query"], "error_type": type(error).__name__})
                reranked = balanced
            final = shared.final_results(reranked, docs)
            direct_final = shared.final_results(lateon_order, docs) if lateon_results else shared.final_results(balanced, docs)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            run_fingerprint.append(final)
            if run == 0:
                records.append({
                    "query": case["query"],
                    "answers_query": case["answers_query"],
                    "metrics": shared.metrics(final, docs, case),
                    "latency_ms": elapsed,
                    "lateon_score_min": min((result_score(item) for item in lateon_results), default=None),
                    "lateon_score_max": max((result_score(item) for item in lateon_results), default=None),
                    "results": [
                        {
                            "file": docs[doc_id]["path"],
                            "start": docs[doc_id]["start"],
                            "end": docs[doc_id]["end"],
                        }
                        for doc_id in final
                    ],
                })
                direct_records.append({
                    "query": case["query"],
                    "answers_query": case["answers_query"],
                    "metrics": shared.metrics(direct_final, docs, case),
                    "results": [
                        {
                            "file": docs[doc_id]["path"],
                            "start": docs[doc_id]["start"],
                            "end": docs[doc_id]["end"],
                        }
                        for doc_id in direct_final
                    ],
                })
        ranking_fingerprints.append(run_fingerprint)

    answered = [record for record in records if record["answers_query"]]
    direct_answered = [record for record in direct_records if record["answers_query"]]
    artifact = {
        "experiment": "isolated LateOn-Code shortlist reranker",
        "model": MODEL,
        "model_revision": REVISION,
        "license": "Apache-2.0",
        "corpus_commit": dataset["corpus_hash"],
        "runs": args.runs,
        "shortlist": 20,
        "fusion": "rank RRF with predeclared production 0.75 baseline / 0.25 reranker weights",
        "model_load_seconds": load_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mrr": sum(record["metrics"]["mrr"] for record in answered) / len(answered),
        "ndcg@10": sum(record["metrics"]["ndcg@10"] for record in answered) / len(answered),
        "recall@10": sum(record["metrics"]["recall@10"] for record in answered) / len(answered),
        "promotion_errors": sum(record["metrics"]["promotion_errors"] for record in answered),
        "direct_order": {
            "mrr": sum(record["metrics"]["mrr"] for record in direct_answered) / len(direct_answered),
            "ndcg@10": sum(record["metrics"]["ndcg@10"] for record in direct_answered) / len(direct_answered),
            "recall@10": sum(record["metrics"]["recall@10"] for record in direct_answered) / len(direct_answered),
            "promotion_errors": sum(record["metrics"]["promotion_errors"] for record in direct_answered),
            "records": direct_records,
        },
        "p50_latency_ms": shared.percentile(latencies, 0.5),
        "p95_latency_ms": shared.percentile(latencies, 0.95),
        "rankings_identical_across_runs": all(
            fingerprint == ranking_fingerprints[0]
            for fingerprint in ranking_fingerprints[1:]
        ),
        "failures": failures,
        "fallback_count": len(failures),
        "answerability": "Unsupported: no calibrated threshold; both no-answer controls return results.",
        "fallback": "The isolated wrapper returns unchanged balanced order on model failure; production was not modified.",
        "privacy": "Local CPU inference only; no query or source content left the orb.",
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
