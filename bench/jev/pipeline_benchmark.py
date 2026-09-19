#!/usr/bin/env python3
"""Isolated benchmark for ranking-pipeline changes without replacing Nomic/Jina."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sqlite3
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np


def load_shared():
    path = Path(__file__).with_name("coderank_benchmark.py")
    spec = importlib.util.spec_from_file_location("coderank_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nomic_query(query: str) -> np.ndarray:
    request = urllib.request.Request(
        "http://localhost:8080/embedding",
        data=json.dumps({"content": "search_query: " + query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    vector = np.asarray(payload[0]["embedding"][0], dtype=np.float32)
    return vector / max(np.linalg.norm(vector), 1e-12)


def fused_order(shared, ids: list[str], docs: dict[str, dict], query: str, scores: list[float]) -> list[str]:
    top = ids[:20]
    if len(scores) != len(top):
        raise ValueError(f"received {len(scores)} reranker scores for {len(top)} candidates")
    reranker = dict(zip(top, scores))
    vector_weight, reranker_weight = 0.75, 0.25
    if shared.intent(query) == "example":
        reranker_weight = 0.35
    total = vector_weight + reranker_weight
    vector_weight, reranker_weight = vector_weight / total, reranker_weight / total
    fused = []
    for position, doc_id in enumerate(ids):
        vector_score = 1 / (61 + position)
        reranker_score = 0.0
        if doc_id in reranker:
            probability = 1 / (1 + math.exp(-reranker[doc_id]))
            reranker_score = 1 / (61 + (1 - probability) * 100)
        boost = shared.code_boost(docs[doc_id]["path"], shared.intent(query))
        fused.append((doc_id, (vector_weight * vector_score + reranker_weight * reranker_score) * boost))
    return [doc_id for doc_id, _ in sorted(fused, key=lambda item: (-item[1], item[0]))]


def eligible_deduplicated(ids: list[str], docs: dict[str, dict]) -> list[str]:
    kept = []
    for doc_id in ids:
        doc = docs[doc_id]
        if doc["is_test"]:
            continue
        if any(docs[seen]["path"] == doc["path"] and overlaps(docs[seen], doc) for seen in kept):
            continue
        kept.append(doc_id)
    return kept


def overlaps(left: dict, right: dict) -> bool:
    overlap = min(left["end"], right["end"]) - max(left["start"], right["start"]) + 1
    if overlap <= 0:
        return False
    return overlap / (left["end"] - left["start"] + 1) > 0.5 or overlap / (right["end"] - right["start"] + 1) > 0.5


def diversify(ids: list[str], docs: dict[str, dict], per_file: int = 2) -> list[str]:
    counts = Counter()
    result = []
    deferred = []
    for doc_id in ids:
        path = docs[doc_id]["path"]
        if counts[path] < per_file:
            counts[path] += 1
            result.append(doc_id)
        else:
            deferred.append(doc_id)
    return result + deferred


def implementation_first(ids: list[str], docs: dict[str, dict]) -> list[str]:
    implementation = []
    prose_or_test = []
    for doc_id in ids:
        path = docs[doc_id]["path"].lower()
        is_prose = (
            path.endswith(".md")
            or path.startswith(("docs/", "examples/", "compatibility_test/"))
            or "/docs/" in path
            or "/examples/" in path
        )
        (prose_or_test if docs[doc_id]["is_test"] or is_prose else implementation).append(doc_id)
    return implementation + prose_or_test


def summarize(shared, records: list[dict], latencies: list[float]) -> dict:
    answered = [record for record in records if record["answers_query"]]
    return {
        "mrr": sum(record["metrics"]["mrr"] for record in answered) / len(answered),
        "ndcg@10": sum(record["metrics"]["ndcg@10"] for record in answered) / len(answered),
        "recall@10": sum(record["metrics"]["recall@10"] for record in answered) / len(answered),
        "promotion_errors": sum(record["metrics"]["promotion_errors"] for record in answered),
        "p50_latency_ms": shared.percentile(latencies, 0.5),
        "p95_latency_ms": shared.percentile(latencies, 0.95),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--dataset", default="bench/jev/dataset.json")
    parser.add_argument("--jina-endpoint", default="http://127.0.0.1:8084")
    parser.add_argument("--output", default="bench/jev/results-pipeline-variants.json")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    shared = load_shared()
    connection = sqlite3.connect(args.database)
    rows = connection.execute(
        "SELECT id, filepath, content, start_line, end_line, metadata, is_test FROM documents ORDER BY rowid"
    ).fetchall()
    docs = {}
    for doc_id, path, content, start, end, metadata, is_test in rows:
        description = (json.loads(metadata or "{}").get("description") or "").strip()
        text = (description + "\n\n" + content.strip()).strip() if description else content.strip()
        docs[doc_id] = {"path": path, "start": start, "end": end, "is_test": bool(is_test), "text": text}

    vector_ids, vector_matrix = shared.load_sgrep_vectors(Path(args.database).parent / "vectors.mmap")
    dataset = json.loads(Path(args.dataset).read_text())
    if "cases" in dataset:
        cases = [case for case in dataset["cases"] if case["split"] == "heldout"]
    else:
        cases = [
            {**case, "answers_query": True}
            for case in dataset["queries"]
        ]
    lexical = {}
    for case in cases:
        terms = shared.hybrid_terms(case["query"])
        lexical[case["query"]] = [row[0] for row in connection.execute(
            "SELECT d.id FROM documents_fts f JOIN documents d ON d.rowid=f.rowid "
            "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) ASC LIMIT 250", (terms,)
        )] if terms else []

    names = [
        "balanced baseline",
        "balanced implementation-first",
        "current Jina boundary",
        "preserve fused order",
        "post-filter fill",
        "preserve order + post-filter fill",
        "prefilter shortlist + preserve/fill",
        "preserve/fill + max two chunks per file",
        "Jina preserve/fill implementation-first",
    ]
    first_records = {name: [] for name in names}
    latencies = {name: [] for name in names}
    fingerprints = {name: [] for name in names}
    failures = 0

    for run in range(args.runs):
        run_fingerprints = {name: [] for name in names}
        for case in cases:
            started = time.perf_counter()
            query_vector = nomic_query(case["query"])
            similarities = vector_matrix @ query_vector
            dense_indices = np.argpartition(-similarities, 249)[:250]
            dense_indices = dense_indices[np.argsort(-similarities[dense_indices])]
            dense = [vector_ids[index] for index in dense_indices if 1 - similarities[index] <= 1.5]
            balanced = shared.rrf(dense, lexical[case["query"]], 50)
            retrieval_ms = (time.perf_counter() - started) * 1000

            baseline = shared.final_results(balanced, docs)
            variants = {
                "balanced baseline": baseline,
                "balanced implementation-first": shared.final_results(implementation_first(balanced, docs), docs),
            }
            variant_latency = {
                "balanced baseline": retrieval_ms,
                "balanced implementation-first": retrieval_ms,
            }
            try:
                rerank_started = time.perf_counter()
                scores = shared.jina_scores(args.jina_endpoint, case["query"], [docs[item]["text"] for item in balanced[:20]])
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
                ordered = fused_order(shared, balanced, docs, case["query"], scores)
                selected = set(ordered[:10])
                variants["current Jina boundary"] = shared.final_results([item for item in balanced if item in selected], docs)
                variants["preserve fused order"] = shared.final_results(ordered[:10], docs)
                variants["post-filter fill"] = shared.final_results(
                    [item for item in balanced if item in selected]
                    + [item for item in balanced if item not in selected],
                    docs,
                )
                variants["preserve order + post-filter fill"] = shared.final_results(ordered, docs)
                variants["preserve/fill + max two chunks per file"] = shared.final_results(diversify(ordered, docs), docs)
                variants["Jina preserve/fill implementation-first"] = shared.final_results(implementation_first(ordered, docs), docs)
                for name in (
                    "current Jina boundary",
                    "preserve fused order",
                    "post-filter fill",
                    "preserve order + post-filter fill",
                    "preserve/fill + max two chunks per file",
                    "Jina preserve/fill implementation-first",
                ):
                    variant_latency[name] = retrieval_ms + rerank_ms

                eligible = eligible_deduplicated(balanced, docs)
                prefilter_started = time.perf_counter()
                prefilter_scores = shared.jina_scores(args.jina_endpoint, case["query"], [docs[item]["text"] for item in eligible[:20]])
                prefilter_ms = (time.perf_counter() - prefilter_started) * 1000
                prefilter_order = fused_order(shared, eligible, docs, case["query"], prefilter_scores)
                variants["prefilter shortlist + preserve/fill"] = shared.final_results(prefilter_order, docs)
                variant_latency["prefilter shortlist + preserve/fill"] = retrieval_ms + prefilter_ms
            except Exception:
                failures += 1
                for name in names[2:]:
                    variants[name] = baseline
                    variant_latency[name] = retrieval_ms

            for name in names:
                final = variants[name]
                latencies[name].append(variant_latency[name])
                run_fingerprints[name].append(final)
                if run == 0:
                    first_records[name].append({
                        "query": case["query"],
                        "answers_query": case["answers_query"],
                        "metrics": shared.metrics(final, docs, case),
                        "result_count": len(final),
                        "results": [{"file": docs[item]["path"], "start": docs[item]["start"], "end": docs[item]["end"]} for item in final],
                    })
        for name in names:
            fingerprints[name].append(run_fingerprints[name])

    arms = []
    for name in names:
        arm = {"name": name, **summarize(shared, first_records[name], latencies[name])}
        arm["rankings_identical_across_runs"] = all(item == fingerprints[name][0] for item in fingerprints[name][1:])
        arms.append(arm)
    artifact = {
        "experiment": "isolated no-model-replacement pipeline variants",
        "dataset": dataset.get("name", Path(args.dataset).name),
        "corpus_commit": dataset["corpus_hash"],
        "runs": args.runs,
        "models": {"retrieval": "existing Nomic", "reranker": "existing Jina v2"},
        "predeclared_variants": names,
        "failures": failures,
        "fallback": "On any Jina request failure, every affected variant returns the unchanged balanced baseline.",
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "privacy": "All inference local; no query or source content left the orb. Artifact excludes source bodies.",
        "limitations": "Fixed labels are not blind-human-audited. Identifier-aware FTS is not scored because this held-out set has no identifier-focused labels.",
        "arms": arms,
    }
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    for arm in arms:
        print(arm["name"], f'{arm["mrr"]:.3f}/{arm["ndcg@10"]:.3f}/{arm["recall@10"]:.3f}', f'errors={arm["promotion_errors"]}')


if __name__ == "__main__":
    main()
