#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Quality comparison: sgrep vs osgrep

Compares search result QUALITY using precision/recall metrics.
Run with: uv run scripts/quality_compare.py
"""

import subprocess
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_PATH = "/Users/xiao/development/github.com/XiaoConstantine/maestro"
SGREP_BIN = "/Users/xiao/development/github.com/XiaoConstantine/sgrep/sgrep"

# Test cases: (query, [expected_files])
# Expected files are ones that SHOULD appear in top results
TEST_CASES = [
    ("embedding generation", 
     ["embedding_router.go", "embedding_cache.go", "multi_vector_embedding.go"]),
    
    ("validate code reviews",
     ["review_processor.go", "review.go", "consensus_validation_processor.go"]),
    
    ("fetch pull request from github",
     ["github_tool.go", "review.go"]),
    
    ("store and retrieve vectors",
     ["rag.go", "embedding_router.go"]),
    
    ("parse go source code into functions",
     ["chunk.go", "context_extractor.go"]),
    
    ("coordinate multiple ai agents",
     ["agentic_orchestrator.go", "agent_spawner.go", "unified_react_agent.go"]),
    
    ("manage claude sessions",
     ["claude_session_manager.go", "claude_coordinator.go"]),
    
    ("calculate text similarity",
     ["similarity_utils.go", "result_aggregator.go"]),
    
    ("process review comments",
     ["comment_processor.go", "comment_refinement_processor.go"]),
    
    ("search code semantically",
     ["search_agent.go", "simple_search.go", "rag.go"]),
]

@dataclass
class SearchResult:
    files: list[str]
    scores: list[float]

def run_sgrep(query: str) -> SearchResult:
    """Run sgrep and extract results."""
    try:
        result = subprocess.run(
            [SGREP_BIN, query, "--json", "-n", "10"],
            capture_output=True, text=True, cwd=REPO_PATH, timeout=30
        )
        if result.returncode != 0:
            return SearchResult([], [])
        
        data = json.loads(result.stdout)
        files = []
        scores = []
        seen = set()
        for item in data:
            basename = os.path.basename(item.get("file", ""))
            if basename and basename not in seen:
                seen.add(basename)
                files.append(basename)
                scores.append(item.get("score", 0))
        return SearchResult(files, scores)
    except Exception as e:
        print(f"  sgrep error: {e}")
        return SearchResult([], [])

def run_osgrep(query: str) -> SearchResult:
    """Run osgrep and extract results."""
    try:
        result = subprocess.run(
            ["osgrep", "search", query, "--json", "-m", "10"],
            capture_output=True, text=True, cwd=REPO_PATH, timeout=60
        )
        if result.returncode != 0:
            return SearchResult([], [])
        
        # osgrep outputs some debug lines before JSON
        stdout = result.stdout
        json_start = stdout.find('{"results"')
        if json_start == -1:
            return SearchResult([], [])
        
        data = json.loads(stdout[json_start:])
        files = []
        scores = []
        seen = set()
        for item in data.get("results", []):
            basename = os.path.basename(item.get("path", ""))
            if basename and basename not in seen:
                seen.add(basename)
                files.append(basename)
                scores.append(item.get("score", 0))
        return SearchResult(files, scores)
    except Exception as e:
        print(f"  osgrep error: {e}")
        return SearchResult([], [])

def calc_precision(returned: list[str], expected: list[str]) -> float:
    """Precision: what fraction of returned results are relevant?"""
    if not returned:
        return 0.0
    hits = sum(1 for f in returned if f in expected)
    return hits / len(returned)

def calc_recall(returned: list[str], expected: list[str]) -> float:
    """Recall: what fraction of expected results were returned?"""
    if not expected:
        return 0.0
    hits = sum(1 for f in expected if f in returned)
    return hits / len(expected)

def calc_overlap(files1: list[str], files2: list[str]) -> int:
    """Count files appearing in both result sets."""
    return len(set(files1) & set(files2))

def main():
    print("=" * 60)
    print("Semantic Search Quality Comparison: sgrep vs osgrep")
    print("=" * 60)
    print(f"Repository: {REPO_PATH}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()
    
    sgrep_precisions = []
    sgrep_recalls = []
    osgrep_precisions = []
    osgrep_recalls = []
    overlaps = []
    
    detailed_results = []
    
    for query, expected in TEST_CASES:
        print(f'Query: "{query}"')
        print(f"  Expected: {', '.join(expected)}")
        
        sgrep_result = run_sgrep(query)
        osgrep_result = run_osgrep(query)
        
        sp = calc_precision(sgrep_result.files, expected)
        sr = calc_recall(sgrep_result.files, expected)
        op = calc_precision(osgrep_result.files, expected)
        ore = calc_recall(osgrep_result.files, expected)
        overlap = calc_overlap(sgrep_result.files, osgrep_result.files)
        
        sgrep_precisions.append(sp)
        sgrep_recalls.append(sr)
        osgrep_precisions.append(op)
        osgrep_recalls.append(ore)
        overlaps.append(overlap)
        
        print(f"  sgrep:  P={sp:.2f} R={sr:.2f} files={sgrep_result.files[:5]}")
        print(f"  osgrep: P={op:.2f} R={ore:.2f} files={osgrep_result.files[:5]}")
        print(f"  Overlap: {overlap} files")
        print()
        
        detailed_results.append({
            "query": query,
            "expected": expected,
            "sgrep": {"files": sgrep_result.files, "precision": sp, "recall": sr},
            "osgrep": {"files": osgrep_result.files, "precision": op, "recall": ore},
            "overlap": overlap
        })
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("| Metric          | sgrep  | osgrep |")
    print("|-----------------|--------|--------|")
    print(f"| Avg Precision   | {sum(sgrep_precisions)/len(sgrep_precisions):.3f}  | {sum(osgrep_precisions)/len(osgrep_precisions):.3f}  |")
    print(f"| Avg Recall      | {sum(sgrep_recalls)/len(sgrep_recalls):.3f}  | {sum(osgrep_recalls)/len(osgrep_recalls):.3f}  |")
    print(f"| Avg Overlap     | {sum(overlaps)/len(overlaps):.1f} files      |")
    print()
    
    # F1 scores
    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    sgrep_f1 = f1(sum(sgrep_precisions)/len(sgrep_precisions), 
                  sum(sgrep_recalls)/len(sgrep_recalls))
    osgrep_f1 = f1(sum(osgrep_precisions)/len(osgrep_precisions),
                   sum(osgrep_recalls)/len(osgrep_recalls))
    
    print(f"F1 Score: sgrep={sgrep_f1:.3f} osgrep={osgrep_f1:.3f}")
    print()
    
    # Technical notes
    print("Technical Notes:")
    print("  sgrep:  nomic-embed-text-v1.5 (768d), llama.cpp, L2 distance")
    print("  osgrep: mxbai-embed-xsmall-v1 (384d), transformers.js, LanceDB")
    print()
    print("Interpretation:")
    print("  - Precision: Of results returned, how many are relevant?")
    print("  - Recall: Of expected results, how many were found?")
    print("  - F1: Harmonic mean of precision and recall")
    print("  - Overlap: Agreement between tools (different ≠ wrong)")
    
    # Save JSON results
    output_path = Path(__file__).parent.parent / "benchmark" / "quality_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "sgrep": {
                    "avg_precision": sum(sgrep_precisions)/len(sgrep_precisions),
                    "avg_recall": sum(sgrep_recalls)/len(sgrep_recalls),
                    "f1": sgrep_f1
                },
                "osgrep": {
                    "avg_precision": sum(osgrep_precisions)/len(osgrep_precisions),
                    "avg_recall": sum(osgrep_recalls)/len(osgrep_recalls),
                    "f1": osgrep_f1
                },
                "avg_overlap": sum(overlaps)/len(overlaps)
            },
            "details": detailed_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
