"""
Utils — Helper functions for formatting, evaluation, and metrics
"""

from typing import List, Dict, Any


def format_sources(sources: List[Dict]) -> str:
    """Format retrieved sources into a readable string."""
    if not sources:
        return "No sources retrieved."
    lines = []
    for i, s in enumerate(sources):
        lines.append(
            f"[{i+1}] {s['source']} (Page {s.get('page', '?')}) "
            f"| Similarity: {s.get('score', 0):.3f}\n"
            f"    ...{s['snippet']}..."
        )
    return "\n".join(lines)


def compute_precision_at_k(retrieved: List[Dict], relevant_keywords: List[str], k: int = 3) -> float:
    """
    Simple keyword-based Precision@K evaluation.
    Checks what fraction of retrieved chunks contain at least one relevant keyword.
    
    In a real system, you'd use a labeled dataset. This is a proxy metric.
    """
    if not retrieved or not relevant_keywords:
        return 0.0
    top_k = retrieved[:k]
    hits = 0
    for chunk in top_k:
        content_lower = chunk["content"].lower()
        if any(kw.lower() in content_lower for kw in relevant_keywords):
            hits += 1
    return hits / len(top_k)


def compute_mrr(retrieved: List[Dict], relevant_keywords: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR).
    Returns 1/rank of the first relevant chunk. 0 if none found.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        content_lower = chunk["content"].lower()
        if any(kw.lower() in content_lower for kw in relevant_keywords):
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(rag_pipeline, test_cases: List[Dict]) -> Dict[str, float]:
    """
    Run evaluation on a list of test cases.
    
    test_cases format:
    [
        {"query": "What is X?", "keywords": ["X", "related term"]},
        ...
    ]
    
    Returns avg Precision@3 and MRR across all test cases.
    """
    p_at_3_scores = []
    mrr_scores = []

    for tc in test_cases:
        retrieved = rag_pipeline.retrieve(tc["query"])
        p3 = compute_precision_at_k(retrieved, tc["keywords"], k=3)
        mrr = compute_mrr(retrieved, tc["keywords"])
        p_at_3_scores.append(p3)
        mrr_scores.append(mrr)

    return {
        "avg_precision_at_3": sum(p_at_3_scores) / len(p_at_3_scores),
        "avg_mrr": sum(mrr_scores) / len(mrr_scores),
        "num_test_cases": len(test_cases)
    }
