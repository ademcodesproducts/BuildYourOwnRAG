"""
Reciprocal Rank Fusion (RRF) for combining dense and BM25 retrieval results.
"""


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Combine dense and BM25 results using reciprocal rank fusion.

    Each chunk's score is the sum of 1/(k + rank + 1) across the two ranked lists.
    Higher score = better combined rank.

    Args:
        dense_results: Ranked list of chunk dicts from dense retrieval.
        bm25_results:  Ranked list of chunk dicts from BM25 retrieval.
        k:      RRF smoothing constant (default 60, standard value from the literature).
        top_k:  Number of top results to return.

    Returns:
        List of up to top_k chunk dicts, re-ranked by fused score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk_map[cid] for cid, _ in ranked]
