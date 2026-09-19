#!/usr/bin/env python3
"""Isolated CodeRankEmbed candidate-generation benchmark.

This script reads sgrep's existing deterministic chunks, stores CodeRankEmbed
vectors outside the repository, and evaluates frozen labels without changing
production indexing or search behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sqlite3
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL = "nomic-ai/CodeRankEmbed"
REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
QUERY_PREFIX = "Represent this query for searching relevant code: "
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "is", "are", "was", "were", "be", "have", "has", "had", "do",
    "does", "did", "how", "what", "where", "when", "why", "which", "find", "show", "get", "list",
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_sgrep_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b"SGVE":
        raise RuntimeError(f"invalid sgrep vector file: {path}")
    version, dimensions, count, data_offset = struct.unpack_from("<IIII", data, 4)
    if version != 1 or dimensions <= 0 or count <= 0:
        raise RuntimeError(f"unsupported sgrep vector header: version={version}, dimensions={dimensions}, count={count}")
    offset = 32
    ids = []
    for _ in range(count):
        (length,) = struct.unpack_from("<H", data, offset)
        offset += 2
        ids.append(data[offset : offset + length].decode())
        offset += length
    if offset > data_offset:
        raise RuntimeError("sgrep vector index overlaps vector data")
    vectors = np.frombuffer(data, dtype="<f4", count=count * dimensions, offset=data_offset)
    return ids, vectors.reshape(count, dimensions)


def hybrid_terms(query: str) -> str:
    terms = []
    for word in query.lower().split():
        word = word.strip(".,?!\"'`:;()[]{}*")
        if len(word) < 2 or word in STOPWORDS:
            continue
        if not all(char == "_" or char.isalnum() for char in word):
            word = '"' + word.replace('"', '""') + '"'
        terms.append(word)
    if len(terms) < 2:
        return terms[0] if terms else ""
    compounds: dict[int, list[str]] = {i: [] for i in range(len(terms))}
    for width in range(2, min(4, len(terms)) + 1):
        for start in range(len(terms) - width + 1):
            window = terms[start : start + width]
            if any('"' in term for term in window):
                continue
            compound = "".join(window)
            for index in range(start, start + width):
                compounds[index].append(compound)
    groups = []
    for index, term in enumerate(terms):
        alternatives = list(dict.fromkeys([term, *compounds[index]]))
        groups.append(term if len(alternatives) == 1 else "(" + " OR ".join(alternatives) + ")")
    return " OR ".join(groups)


def rrf(dense: list[str], lexical: list[str], limit: int, semantic_weight: float = 0.6, bm25_weight: float = 0.4) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dict.fromkeys(dense)):
        scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight / (60 + rank + 1)
    for rank, doc_id in enumerate(dict.fromkeys(lexical)):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (60 + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def intent(query: str) -> str:
    if re.search(r"\b(example|sample|demo|usage|show\s+me)\b", query, re.I):
        return "example"
    if re.search(r"\b(how\s+to|implement|handle|process|manage|create|make|build)\b", query, re.I):
        return "procedural"
    if re.search(r"\b(type|interface|struct|class|enum|definition|define|what\s+is)\b", query, re.I):
        return "definition"
    return "conceptual"


def code_boost(path: str, query_intent: str) -> float:
    path = path.lower()
    boost = 1.0
    if query_intent == "conceptual":
        if path in {"readme.md", "readme", "readme.txt"}:
            boost *= 1.5
        elif path.endswith("/readme.md") and "example" not in path:
            boost *= 1.2
    if query_intent in {"conceptual", "definition"}:
        if path.startswith(("pkg/", "internal/", "lib/")) and "example" not in path and "test" not in path:
            boost *= 1.3
        if "/example" in path or path.startswith("example"):
            boost *= 0.7
    if query_intent == "example" and ("example" in path or "demo" in path):
        boost *= 1.3
    if path.endswith("_test.go") or "/test/" in path:
        boost *= 0.8
    if ".worktrees/" in path:
        boost *= 0.5
    if ".generated." in path or ".pb.go" in path:
        boost *= 0.6
    return boost


def jina_scores(endpoint: str, query: str, documents: list[str]) -> list[float]:
    body = json.dumps({"query": query, "documents": documents}).encode()
    request = urllib.request.Request(endpoint + "/v1/rerank", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    scores = [float("-inf")] * len(documents)
    for result in payload["results"]:
        scores[result["index"]] = result["relevance_score"]
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("Jina returned an incomplete ranking")
    return scores


def fuse_reranker_scores(ids: list[str], docs: dict[str, dict], query: str, scores: list[float]) -> list[str]:
    top = ids[:20]
    if len(scores) != len(top):
        raise ValueError(f"received {len(scores)} reranker scores for {len(top)} candidates")
    reranker = {doc_id: score for doc_id, score in zip(top, scores)}
    vector_weight, reranker_weight = (0.75, 0.25)
    if intent(query) == "example":
        reranker_weight = 0.35
    total = vector_weight + reranker_weight
    vector_weight, reranker_weight = vector_weight / total, reranker_weight / total
    fused = []
    for rank, doc_id in enumerate(ids):
        vector_score = 1 / (60 + rank + 1)
        reranker_score = 0.0
        if doc_id in reranker:
            probability = 1 / (1 + math.exp(-reranker[doc_id]))
            reranker_score = 1 / (60 + (1 - probability) * 100 + 1)
        score = (vector_weight * vector_score + reranker_weight * reranker_score) * code_boost(docs[doc_id]["path"], intent(query))
        fused.append((doc_id, score))
    # Searcher.rerankResults selects opts.Limit by fused score, then
    # SearchWithOptions restores the selected documents' original-distance
    # order before filtering and deduplication.
    selected = {doc_id for doc_id, _ in sorted(fused, key=lambda item: -item[1])[:10]}
    return [doc_id for doc_id in ids if doc_id in selected]


def rerank(ids: list[str], docs: dict[str, dict], query: str, endpoint: str) -> list[str]:
    top = ids[:20]
    scores = jina_scores(endpoint, query, [docs[doc_id]["text"] for doc_id in top])
    return fuse_reranker_scores(ids, docs, query, scores)


def overlaps(left: dict, right: dict) -> bool:
    overlap_start = max(left["start"], right["start"])
    overlap_end = min(left["end"], right["end"])
    overlap_lines = overlap_end - overlap_start + 1
    if overlap_lines <= 0:
        return False
    left_lines = left["end"] - left["start"] + 1
    right_lines = right["end"] - right["start"] + 1
    return overlap_lines / left_lines > 0.5 or overlap_lines / right_lines > 0.5


def final_results(ids: list[str], docs: dict[str, dict], limit: int = 10) -> list[str]:
    results = []
    for doc_id in ids:
        doc = docs[doc_id]
        if doc["is_test"]:
            continue
        if any(docs[seen]["path"] == doc["path"] and overlaps(docs[seen], doc) for seen in results):
            continue
        results.append(doc_id)
        if len(results) == limit:
            break
    return results


def metrics(ids: list[str], docs: dict[str, dict], case: dict) -> dict:
    judgments = {item["file"]: item["rel"] for item in case["judgments"]}
    ideal = sorted(judgments.values(), reverse=True)
    seen_paths = set()
    ranked = []
    first = None
    prose_before = 0
    relevant_seen = 0
    reciprocal_rank = 0.0
    for doc_id in ids:
        path = docs[doc_id]["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        relevance = judgments.get(path, 0)
        rank = len(ranked)
        ranked.append(relevance)
        if relevance:
            relevant_seen += 1
            if reciprocal_rank == 0:
                reciprocal_rank = 1 / (rank + 1)
            if first is None:
                first = rank
        elif first is None and (path.endswith((".md", "_test.go")) or path.startswith(("docs/", "examples/", "compatibility_test/")) or "/docs/" in path or "/examples/" in path):
            prose_before += 1
        if len(ranked) == 10:
            break
    def dcg(values: list[int]) -> float:
        return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(values[:10]))
    return {
        "mrr": reciprocal_rank,
        "ndcg@10": dcg(ranked) / dcg(ideal) if ideal else 0,
        "recall@10": relevant_seen / len(judgments) if judgments else 0,
        "promotion_errors": prose_before if first is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--dataset", default="bench/jev/dataset.json")
    parser.add_argument("--cache", default=str(Path.home() / ".cache/sgrep-coderank"))
    parser.add_argument("--jina-endpoint", default="http://127.0.0.1:8084")
    parser.add_argument("--output", default="bench/jev/results-coderankembed.json")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.database)
    rows = connection.execute("SELECT id, filepath, content, start_line, end_line, metadata, is_test FROM documents ORDER BY rowid").fetchall()
    docs = {}
    ids = []
    texts = []
    for doc_id, path, content, start, end, metadata, is_test in rows:
        description = (json.loads(metadata or "{}").get("description") or "").strip()
        text = (description + "\n\n" + content.strip()).strip() if description else content.strip()
        docs[doc_id] = {"path": path, "start": start, "end": end, "is_test": bool(is_test), "text": text}
        ids.append(doc_id)
        texts.append(text)
    nomic_ids, nomic_matrix = load_sgrep_vectors(Path(args.database).parent / "vectors.mmap")
    if set(nomic_ids) != set(ids):
        raise RuntimeError("sgrep vector IDs do not match SQLite documents")

    model = SentenceTransformer(MODEL, revision=REVISION, trust_remote_code=True, device="cpu")
    vector_path = cache / f"coderank-{REVISION[:12]}-{len(ids)}.npy"
    index_started = time.perf_counter()
    if vector_path.exists():
        coderank_matrix = np.load(vector_path)
        indexed = False
    else:
        coderank_matrix = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(vector_path, coderank_matrix)
        indexed = True
    index_seconds = time.perf_counter() - index_started

    dataset = json.loads(Path(args.dataset).read_text())
    cases = [case for case in dataset["cases"] if case["split"] == "heldout"]
    lexical = {}
    for case in cases:
        terms = hybrid_terms(case["query"])
        lexical[case["query"]] = [row[0] for row in connection.execute(
            "SELECT d.id FROM documents_fts f JOIN documents d ON d.rowid=f.rowid WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) ASC LIMIT 250",
            (terms,),
        )] if terms else []

    arms = {name: [] for name in ("Nomic balanced", "Nomic balanced + Jina", "CodeRankEmbed dense", "CodeRankEmbed balanced", "CodeRankEmbed balanced + Jina")}
    candidate_recall = {name: [] for name in arms}
    latencies = {name: [] for name in arms}
    for run in range(args.runs):
        for case in cases:
            query = case["query"]
            for family, matrix, matrix_ids in (
                ("Nomic", nomic_matrix, nomic_ids),
                ("CodeRankEmbed", coderank_matrix, ids),
            ):
                started = time.perf_counter()
                if family == "Nomic":
                    body = json.dumps({"content": "search_query: " + query}).encode()
                    request = urllib.request.Request("http://localhost:8080/embedding", data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = json.load(response)
                    query_vector = np.asarray(payload[0]["embedding"][0], dtype=np.float32)
                    query_vector /= max(np.linalg.norm(query_vector), 1e-12)
                else:
                    query_vector = model.encode(
                        [QUERY_PREFIX + query],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )[0].astype(np.float32)
                similarities = matrix @ query_vector
                dense_indices = np.argpartition(-similarities, 249)[:250]
                dense_indices = dense_indices[np.argsort(-similarities[dense_indices])]
                dense = [matrix_ids[index] for index in dense_indices if 1 - similarities[index] <= 1.5]
                balanced = rrf(dense, lexical[query], 50)
                if family == "CodeRankEmbed":
                    dense_final = final_results(dense[:50], docs)
                    elapsed = (time.perf_counter() - started) * 1000
                    record = {"query": query, "metrics": metrics(dense_final, docs, case), "results": [{"file": docs[item]["path"], "start": docs[item]["start"], "end": docs[item]["end"]} for item in dense_final]}
                    if run == 0:
                        arms["CodeRankEmbed dense"].append(record)
                    latencies["CodeRankEmbed dense"].append(elapsed)
                    if run == 0 and case["answers_query"]:
                        relevant = {item["file"] for item in case["judgments"]}
                        candidate_recall["CodeRankEmbed dense"].append(len(relevant & {docs[item]["path"] for item in dense[:50]}) / len(relevant))
                balanced_final = final_results(balanced, docs)
                elapsed = (time.perf_counter() - started) * 1000
                arm_name = family + " balanced"
                if run == 0:
                    arms[arm_name].append({"query": query, "metrics": metrics(balanced_final, docs, case), "results": [{"file": docs[item]["path"], "start": docs[item]["start"], "end": docs[item]["end"]} for item in balanced_final]})
                latencies[arm_name].append(elapsed)
                if run == 0 and case["answers_query"]:
                    relevant = {item["file"] for item in case["judgments"]}
                    candidate_recall[arm_name].append(len(relevant & {docs[item]["path"] for item in balanced}) / len(relevant))

                rerank_started = time.perf_counter()
                reranked = rerank(balanced, docs, query, args.jina_endpoint)
                reranked_final = final_results(reranked, docs)
                rerank_elapsed = elapsed + (time.perf_counter() - rerank_started) * 1000
                rerank_name = family + " balanced + Jina"
                if run == 0:
                    arms[rerank_name].append({"query": query, "metrics": metrics(reranked_final, docs, case), "results": [{"file": docs[item]["path"], "start": docs[item]["start"], "end": docs[item]["end"]} for item in reranked_final]})
                latencies[rerank_name].append(rerank_elapsed)
                if run == 0 and case["answers_query"]:
                    candidate_recall[rerank_name].append(candidate_recall[arm_name][-1])

    summaries = []
    for name, records in arms.items():
        answered = [record for record, case in zip(records, cases) if case["answers_query"]]
        summary = {
            "name": name,
            "runs": args.runs,
            "mrr": sum(record["metrics"]["mrr"] for record in answered) / len(answered),
            "ndcg@10": sum(record["metrics"]["ndcg@10"] for record in answered) / len(answered),
            "recall@10": sum(record["metrics"]["recall@10"] for record in answered) / len(answered),
            "promotion_errors": sum(record["metrics"]["promotion_errors"] for record in answered),
            "candidate_recall@50": sum(candidate_recall[name]) / len(candidate_recall[name]),
            "p50_latency_ms": percentile(latencies[name], 0.5),
            "p95_latency_ms": percentile(latencies[name], 0.95),
            "records": records,
        }
        summaries.append(summary)

    artifact = {
        "experiment": "isolated CodeRankEmbed candidate generation",
        "model": MODEL,
        "model_revision": REVISION,
        "query_prefix": QUERY_PREFIX,
        "corpus_commit": dataset["corpus_hash"],
        "documents": len(ids),
        "index_cache": str(vector_path),
        "index_created_this_run": indexed,
        "index_seconds": index_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "labels": "Frozen fixed labels; not human-audited; no model used as judge.",
        "privacy": "Local CPU inference only; model download aside, no query or source content left the orb.",
        "arms": summaries,
    }
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    for summary in summaries:
        print(summary["name"], f'MRR={summary["mrr"]:.3f}', f'NDCG={summary["ndcg@10"]:.3f}', f'R10={summary["recall@10"]:.3f}', f'errors={summary["promotion_errors"]}', f'p50/p95={summary["p50_latency_ms"]:.1f}/{summary["p95_latency_ms"]:.1f}ms')


if __name__ == "__main__":
    main()
