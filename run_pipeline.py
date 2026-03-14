"""
End-to-end RAG pipeline: questions → retrieval → generation → predictions.

Usage:
    python3 run_pipeline.py <questions_path> <predictions_path>
    python3 run_pipeline.py questions.txt predictions.txt
    python3 run_pipeline.py questions.txt predictions.txt --retriever hybrid --top-k 5
"""

import argparse
import logging
import sys
import time

import config
from retriever.dense_retriever import DenseRetriever
from llms.llm_pipeline import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_predictions(path: str, predictions: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Combine dense and BM25 results using reciprocal rank fusion."""
    scores = {}
    chunk_map = {}

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


def build_retriever(method: str, top_k: int):
    """Build and return a retrieval function based on the chosen method."""
    if method == "dense":
        retriever = DenseRetriever()
        retriever.load_index()
        return lambda q: retriever.retrieve_top_k(q, k=top_k)

    elif method == "hybrid":
        from retriever.bm25_retriever import BM25Retriever

        dense = DenseRetriever()
        dense.load_index()
        bm25 = BM25Retriever()
        bm25.load_bm25()

        def hybrid_retrieve(q):
            dense_results = dense.retrieve_top_k(q, k=top_k)
            bm25_results = bm25.retrieve_top_k(q, k=top_k)
            return reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)

        return hybrid_retrieve

    else:
        raise ValueError(f"Unknown retriever method: {method}")


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline: questions → predictions")
    parser.add_argument("questions_path", help="Path to input questions file (one per line)")
    parser.add_argument("predictions_path", help="Path to write predictions (one per line)")
    parser.add_argument("--retriever", default="dense", choices=["dense", "hybrid"],
                        help="Retrieval method (default: dense)")
    parser.add_argument("--top-k", type=int, default=config.DENSE_TOP_K,
                        help="Number of passages to retrieve per question")
    args = parser.parse_args()

    questions = load_questions(args.questions_path)
    logger.info("Loaded %d questions from %s", len(questions), args.questions_path)

    logger.info("Building %s retriever...", args.retriever)
    retrieve = build_retriever(args.retriever, args.top_k)
    logger.info("Retriever ready.")

    predictions = []
    t0 = time.time()

    for i, question in enumerate(questions):
        passages = retrieve(question)
        answer = generate_answer(question, passages)
        predictions.append(answer)

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            logger.info("[%d/%d] %.1fs elapsed, %.2fs/question — last answer: %s",
                        i + 1, len(questions), elapsed, avg, answer)

    write_predictions(args.predictions_path, predictions)
    logger.info("Wrote %d predictions to %s", len(predictions), args.predictions_path)

    total = time.time() - t0
    logger.info("Done in %.1fs (%.2fs/question avg)", total, total / len(questions))


if __name__ == "__main__":
    main()
