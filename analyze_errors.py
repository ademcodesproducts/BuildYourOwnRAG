"""
Error analysis for RAG mispredictions.

For each question, categorizes why the prediction was wrong:
  CORPUS_GAP      — answer not found in any corpus chunk (need more data)
  RETRIEVAL_MISS  — answer in corpus but not retrieved in top-k (improve ranking)
  GENERATION_FAIL — correct context retrieved but LLM still wrong (improve prompt/model)
  FORMAT_MISMATCH — F1 > 0 but EM = 0, partially correct (improve postprocessing)
  FALLBACK        — prediction is "Unknown" or empty (pipeline/API failure)
  CORRECT         — EM = 1

Usage:
    python3 analyze_errors.py
    python3 analyze_errors.py --retriever dense --top-k 10
    python3 analyze_errors.py --output-json results.json --output-csv results.csv
    python3 analyze_errors.py --llm-judge          # also run semantic LLM judge
"""

import argparse
import csv
import json
import logging
import sys
import warnings

import config
from llm import call_llm
from retriever.dense_retriever import DenseRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.fusion import reciprocal_rank_fusion
from run_evaluation import normalize, exact_match, token_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Category constants ────────────────────────────────────────────────────────
CORRECT = "CORRECT"
CORPUS_GAP = "CORPUS_GAP"
RETRIEVAL_MISS = "RETRIEVAL_MISS"
GENERATION_FAIL = "GENERATION_FAIL"
FORMAT_MISMATCH = "FORMAT_MISMATCH"
FALLBACK = "FALLBACK"
UNEVALUATED = "UNEVALUATED"

CATEGORY_ORDER = [CORRECT, FORMAT_MISMATCH, GENERATION_FAIL, RETRIEVAL_MISS, CORPUS_GAP, FALLBACK, UNEVALUATED]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_questions(path: str) -> list[str]:
    """Load questions from a text file, one per line, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_reference_answers(path: str) -> dict[str, list[str]]:
    """Load reference_answers.json and normalise every value to list[str].

    The JSON may have string or list-of-string values. Wrapping everything in
    a list lets all downstream code use a uniform interface.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items()}


def load_predictions(path: str) -> list[str]:
    """Load predictions from a text file, one per line (including blank lines)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_corpus_chunks(path: str) -> list[dict]:
    """Load all chunks from a JSONL file."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d corpus chunks from %s", len(chunks), path)
    return chunks


# ── Corpus preprocessing ──────────────────────────────────────────────────────

def build_normalized_corpus(chunks: list[dict]) -> list[str]:
    """Return a list of normalize(chunk['text']) for every chunk.

    Pre-computing this once avoids repeating normalize() on the same text for
    every question. Empty strings (after normalization) are kept as empty
    strings; callers should skip them when doing substring checks.
    """
    return [normalize(c["text"]) for c in chunks]


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Return (max_em, max_f1) across all reference answers.

    Matches the aggregation strategy used in run_evaluation.py so scores are
    directly comparable.
    """
    em = max(exact_match(prediction, r) for r in references)
    f1 = max(token_f1(prediction, r) for r in references)
    return em, f1


# ── Categorization helpers ────────────────────────────────────────────────────

def _is_fallback(prediction: str) -> bool:
    """Return True if the prediction is empty, whitespace-only, or 'unknown'."""
    return normalize(prediction) == "" or prediction.strip().lower() == "unknown"


def is_answer_in_corpus(
    references: list[str],
    norm_corpus: list[str],
) -> tuple[bool, str | None, int | None]:
    """Check whether any normalized reference is a substring of any normalized chunk.

    Returns:
        (found, matched_ref, matched_chunk_idx)
        found            — True if at least one reference matched
        matched_ref      — the reference string that matched (or None)
        matched_chunk_idx — index into norm_corpus of the first match (or None)
    """
    for ref in references:
        norm_ref = normalize(ref)
        if not norm_ref:
            warnings.warn(
                f"Reference answer '{ref!r}' is empty after normalization — skipping corpus check.",
                stacklevel=2,
            )
            continue
        for idx, norm_chunk in enumerate(norm_corpus):
            if not norm_chunk:
                # Skip chunks that are empty after normalization (would vacuously match)
                continue
            if norm_ref in norm_chunk:
                return True, ref, idx
    return False, None, None


def is_answer_in_retrieved(
    references: list[str],
    retrieved_chunks: list[dict],
    f1_threshold: float = 0.5,
) -> tuple[bool, str | None, int | None]:
    """Check whether any reference is present in any retrieved chunk's text.

    Uses substring match first (fast, exact), then falls back to token F1 >=
    f1_threshold to catch minor variations like "April 18th" vs "April 18" or
    "M.S." vs "Master of Science" that break the substring check but still give
    the LLM enough signal to answer correctly.

    Returns:
        (found, matched_ref, rank)
        found       — True if at least one reference was found
        matched_ref — the reference that matched (or None)
        rank        — 0-indexed position of the matching chunk in retrieved_chunks (or None)
    """
    for ref in references:
        norm_ref = normalize(ref)
        if not norm_ref:
            continue
        for rank, chunk in enumerate(retrieved_chunks):
            norm_chunk = normalize(chunk.get("text", ""))
            if not norm_chunk:
                continue
            # Exact substring match (fast path)
            if norm_ref in norm_chunk:
                return True, ref, rank
            # Fuzzy fallback: high token F1 between reference and chunk
            if token_f1(ref, chunk.get("text", "")) >= f1_threshold:
                return True, ref, rank
    return False, None, None


def _find_best_rank_large(
    references: list[str],
    question: str,
    dense: DenseRetriever,
    bm25: BM25Retriever | None,
    large_k: int = 100,
) -> dict:
    """For RETRIEVAL_MISS: retrieve a larger set and find where the answer ranks.

    This tells us whether the answer was just outside the top-k (easy fix) or
    much further down (harder problem).

    Returns a dict with keys: best_rank_dense, best_rank_hybrid (int or None).
    """
    dense_large = dense.batch_retrieve_top_k([question], k=large_k)[0]
    _, _, dense_rank = is_answer_in_retrieved(references, dense_large)

    hybrid_rank = None
    if bm25 is not None:
        bm25_large = bm25.retrieve_top_k(question, k=large_k)
        fused_large = reciprocal_rank_fusion(dense_large, bm25_large, top_k=large_k)
        _, _, hybrid_rank = is_answer_in_retrieved(references, fused_large)

    return {
        "best_rank_dense": dense_rank,  # None means not in top-100
        "best_rank_hybrid": hybrid_rank,
    }


def _best_partial_f1_in_corpus(references: list[str], chunks: list[dict]) -> float:
    """For CORPUS_GAP: find the highest token F1 between any reference and any chunk.

    A non-zero value here means something semantically related IS in the corpus
    (e.g. a paraphrase), even though the exact substring wasn't found.
    """
    best = 0.0
    for chunk in chunks:
        for ref in references:
            score = token_f1(chunk.get("text", ""), ref)
            if score > best:
                best = score
    return best


# ── Main categorization ───────────────────────────────────────────────────────

def categorize(
    idx: int,
    question: str,
    references: list[str],
    prediction: str,
    norm_corpus: list[str],
    chunks: list[dict],
    dense: DenseRetriever,
    bm25: BM25Retriever | None,
    top_k: int,
) -> dict:
    """Run the full decision tree for one question and return a result dict."""
    em, f1 = compute_scores(prediction, references)

    base = {
        "idx": idx,
        "question": question,
        "references": references,
        "prediction": prediction,
        "em": em,
        "f1": f1,
    }

    # ── Correct ────────────────────────────────────────────────────────────────
    if em == 1.0:
        return {**base, "category": CORRECT, "category_reason": "Exact match."}

    # ── Corpus gap check ───────────────────────────────────────────────────────
    in_corpus, matched_ref, _ = is_answer_in_corpus(references, norm_corpus)

    if not in_corpus:
        best_f1 = _best_partial_f1_in_corpus(references, chunks)
        return {
            **base,
            "category": CORPUS_GAP,
            "category_reason": (
                "Reference answer not found as a substring in any corpus chunk. "
                f"Best partial F1 in corpus: {best_f1:.3f}."
            ),
            "best_partial_f1_in_corpus": round(best_f1, 4),
        }

    # ── Retrieve top-k and check ───────────────────────────────────────────────
    dense_results = dense.batch_retrieve_top_k([question], k=top_k)[0]
    if bm25 is not None:
        bm25_results = bm25.retrieve_top_k(question, k=top_k)
        retrieved = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)
    else:
        retrieved = dense_results

    in_retrieved, _, ret_rank = is_answer_in_retrieved(references, retrieved)

    # ── Retrieval miss ────────────────────────────────────────────────────────
    if not in_retrieved:
        rank_info = _find_best_rank_large(references, question, dense, bm25)
        answer_chunk_ids = [
            chunks[i]["chunk_id"]
            for i, nc in enumerate(norm_corpus)
            if nc and any(normalize(r) in nc for r in references if normalize(r))
        ]
        return {
            **base,
            "category": RETRIEVAL_MISS,
            "category_reason": (
                f"Answer found in corpus but not in top-{top_k} retrieved chunks. "
                f"Best rank in top-100: dense={rank_info['best_rank_dense']}, "
                f"hybrid={rank_info['best_rank_hybrid']}."
            ),
            "answer_chunk_ids": answer_chunk_ids[:10],  # cap to avoid huge output
            **rank_info,
        }

    # ── Answer was retrieved — distinguish remaining categories ───────────────
    top_chunk_ids = [c["chunk_id"] for c in retrieved]
    context_f1 = max(
        token_f1(c.get("text", ""), ref)
        for c in retrieved
        for ref in references
    )

    # FALLBACK: prediction is empty / "Unknown"
    if _is_fallback(prediction):
        return {
            **base,
            "category": FALLBACK,
            "category_reason": (
                f"Correct context was retrieved (rank {ret_rank}) but the model "
                "returned an empty or fallback answer. Likely a pipeline/API failure."
            ),
            "top_retrieved_chunk_ids": top_chunk_ids,
            "context_f1": round(context_f1, 4),
            "raw_prediction": prediction,
        }

    # FORMAT_MISMATCH: some token overlap but not exact
    if f1 > 0.0:
        return {
            **base,
            "category": FORMAT_MISMATCH,
            "category_reason": (
                f"Partial token overlap (F1={f1:.3f}) but not exact match. "
                "Likely a formatting or postprocessing issue."
            ),
            "norm_prediction": normalize(prediction),
            "norm_reference": normalize(matched_ref or references[0]),
            "top_retrieved_chunk_ids": top_chunk_ids,
        }

    # GENERATION_FAIL: context retrieved, prediction non-empty, zero overlap
    return {
        **base,
        "category": GENERATION_FAIL,
        "category_reason": (
            f"Correct context retrieved (rank {ret_rank}, context F1={context_f1:.3f}) "
            "but LLM produced a completely wrong answer."
        ),
        "top_retrieved_chunk_ids": top_chunk_ids,
        "context_f1": round(context_f1, 4),
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def build_summary_table(results: list[dict]) -> str:
    """Return a console-friendly summary table of category counts and %."""
    total = len(results)
    counts: dict[str, int] = {cat: 0 for cat in CATEGORY_ORDER}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    lines = [
        "",
        f"{'Category':<18}  {'Count':>5}  {'%':>6}  Description",
        "-" * 72,
    ]
    descriptions = {
        CORRECT:         "EM = 1",
        FORMAT_MISMATCH: "F1 > 0 but EM = 0 — formatting/postprocessing issue",
        GENERATION_FAIL: "Context retrieved, LLM still wrong — improve prompt/model",
        RETRIEVAL_MISS:  "Answer in corpus but not retrieved — improve ranking",
        CORPUS_GAP:      "Answer absent from corpus — add more data",
        FALLBACK:        "Prediction is 'Unknown'/empty — pipeline/API failure",
        UNEVALUATED:     "No prediction available",
    }
    for cat in CATEGORY_ORDER:
        n = counts.get(cat, 0)
        pct = 100 * n / total if total else 0.0
        lines.append(f"{cat:<18}  {n:>5}  {pct:>5.1f}%  {descriptions[cat]}")
    lines += [
        "-" * 72,
        f"{'TOTAL':<18}  {total:>5}",
    ]

    # If LLM judge was run, append a semantic-accuracy summary
    judged = [r for r in results if r.get("llm_correct") is not None]
    if judged:
        llm_correct = sum(1 for r in judged if r["llm_correct"])
        base_correct = counts.get(CORRECT, 0)
        lines += [
            "",
            "  LLM semantic judge results (FORMAT_MISMATCH + GENERATION_FAIL only):",
            f"  Judged: {len(judged)}  |  LLM-correct: {llm_correct}  |  "
            f"Adjusted accuracy: {(base_correct + llm_correct) / total * 100:.1f}%  "
            f"(vs EM accuracy: {base_correct / total * 100:.1f}%)",
        ]

    lines.append("")
    return "\n".join(lines)


def write_json_report(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON report to %s", path)


def write_csv_report(results: list[dict], path: str) -> None:
    """Write per-question results to CSV.

    Only scalar fields are written; list fields (like chunk_ids) are
    serialized to JSON strings so the CSV stays flat and readable.
    """
    if not results:
        return
    # Collect all keys, ensuring a stable base order
    base_keys = ["idx", "question", "references", "prediction", "em", "f1", "category", "category_reason"]
    extra_keys = sorted({k for r in results for k in r if k not in base_keys})
    fieldnames = base_keys + extra_keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {}
            for k in fieldnames:
                v = r.get(k, "")
                # Serialize lists to JSON strings so they fit in a CSV cell
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
            writer.writerow(row)
    logger.info("Wrote CSV report to %s", path)


def print_examples(results: list[dict], n: int = 3) -> None:
    """Print up to n examples per non-CORRECT category for quick inspection."""
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["category"] != CORRECT:
            by_cat[r["category"]].append(r)

    for cat in CATEGORY_ORDER:
        examples = by_cat.get(cat, [])
        if not examples:
            continue
        print(f"\n{'─' * 60}")
        print(f"  {cat}  ({len(examples)} total — showing up to {n})")
        print(f"{'─' * 60}")
        for ex in examples[:n]:
            print(f"  Q:    {ex['question']}")
            print(f"  Ref:  {ex['references']}")
            print(f"  Pred: {ex['prediction']!r}  (EM={ex['em']:.0f}, F1={ex['f1']:.3f})")
            print(f"  Why:  {ex['category_reason']}")
            if "llm_correct" in ex and ex["llm_correct"] is not None:
                verdict = "CORRECT" if ex["llm_correct"] else "WRONG"
                print(f"  LLM:  {verdict} — {ex.get('llm_judge_reason', '')}")
            print()


# ── LLM judge ────────────────────────────────────────────────────────────────

def run_llm_judge(results: list[dict], model: str = config.LLM_MODEL) -> None:
    """Ask an LLM whether FORMAT_MISMATCH and GENERATION_FAIL predictions are
    semantically correct, even if they don't match the reference string exactly.

    All candidates are sent in a single LLM call requesting a JSON array back.
    Each matching result dict is updated in-place with:
        llm_correct      (bool)  — True if the LLM considers it correct
        llm_judge_reason (str)   — one-sentence explanation

    CORPUS_GAP, RETRIEVAL_MISS, FALLBACK, and CORRECT are skipped — semantic
    judgment only matters when the model produced a real answer but EM/F1 missed it.
    """
    candidates = [r for r in results if r["category"] in (FORMAT_MISMATCH, GENERATION_FAIL)]
    if not candidates:
        logger.info("LLM judge: no FORMAT_MISMATCH or GENERATION_FAIL entries — skipping.")
        return

    logger.info("LLM judge: evaluating %d candidates in one call...", len(candidates))

    # Build a numbered list of items for the LLM to evaluate
    items = []
    for r in candidates:
        ref_str = r["references"][0] if r["references"] else "(none)"
        items.append(
            f'[{r["idx"]}] Question: {r["question"]}\n'
            f'    Reference: {ref_str}\n'
            f'    Prediction: {r["prediction"]}'
        )

    system_prompt = (
        "You are evaluating a factoid QA system for UC Berkeley EECS. "
        "For each numbered item, decide whether the prediction is semantically "
        "equivalent to the reference answer. "
        "Count these as CORRECT: abbreviations (M.S. = Master of Science), "
        "reordered words, minor extra/missing words that don't change meaning, "
        "different but equivalent phrasings. "
        "Count as INCORRECT: different facts, wrong numbers, wrong names. "
        "Respond with ONLY a valid JSON array — no markdown, no extra text."
    )

    query = (
        "For each item below, output {\"idx\": <idx>, \"correct\": true/false, "
        "\"reason\": \"<one sentence>\"}.\n"
        "Return a JSON array containing one object per item.\n\n"
        + "\n\n".join(items)
    )

    # Estimate tokens needed: ~30 chars per entry in the JSON response
    max_response_tokens = max(256, len(candidates) * 40)

    try:
        raw = call_llm(
            query=query,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_response_tokens,
            temperature=0.0,
            timeout=60,
        )
    except RuntimeError as e:
        logger.warning("LLM judge call failed: %s — skipping judge step.", e)
        return

    # Strip markdown code fences if the model wrapped the JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        judgements = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("LLM judge returned invalid JSON (%s) — skipping judge step.\nRaw: %s", e, raw[:200])
        return

    # Build idx → judgement lookup and update each result in-place
    lookup = {j["idx"]: j for j in judgements if isinstance(j, dict) and "idx" in j}
    missing = 0
    for r in candidates:
        j = lookup.get(r["idx"])
        if j is not None:
            r["llm_correct"] = bool(j.get("correct", False))
            r["llm_judge_reason"] = str(j.get("reason", ""))
        else:
            r["llm_correct"] = None   # LLM didn't return a verdict for this idx
            r["llm_judge_reason"] = ""
            missing += 1

    if missing:
        logger.warning("LLM judge: missing verdicts for %d/%d candidates.", missing, len(candidates))

    n_correct = sum(1 for r in candidates if r.get("llm_correct") is True)
    logger.info("LLM judge: %d/%d candidates judged semantically correct.", n_correct, len(candidates))


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Categorize RAG mispredictions.")
    p.add_argument("--questions",    default="questions.txt",           help="Questions file")
    p.add_argument("--references",   default="reference_answers.json",  help="Reference answers JSON")
    p.add_argument("--predictions",  default="predictions.txt",          help="Predictions file")
    p.add_argument("--chunks",       default=config.CHUNKS_JSONL_PATH,  help="Corpus chunks JSONL")
    p.add_argument("--retriever",    default="hybrid", choices=["hybrid", "dense"],
                   help="Retrieval method to use when checking top-k (default: hybrid)")
    p.add_argument("--top-k",        type=int, default=config.DENSE_TOP_K,
                   help="Top-k passages to retrieve per question (default from config)")
    p.add_argument("--output-json",  default="errors_report.json",      help="JSON output path")
    p.add_argument("--output-csv",   default="errors_report.csv",       help="CSV output path")
    p.add_argument("--examples",     type=int, default=3,
                   help="Number of examples per category to print (0 to skip)")
    p.add_argument("--llm-judge",    action="store_true", default=False,
                   help="Run LLM semantic judge on FORMAT_MISMATCH and GENERATION_FAIL entries")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    questions = load_questions(args.questions)
    ref_answers = load_reference_answers(args.references)
    predictions = load_predictions(args.predictions)

    n_q = len(questions)
    n_p = len(predictions)
    if n_p != n_q:
        logger.warning(
            "%d predictions but %d questions — evaluating first %d entries; "
            "remaining questions will be marked UNEVALUATED.",
            n_p, n_q, min(n_p, n_q),
        )

    chunks = load_corpus_chunks(args.chunks)

    # ── Pre-normalise corpus (done once) ───────────────────────────────────────
    logger.info("Pre-normalising corpus...")
    norm_corpus = build_normalized_corpus(chunks)

    # ── Build retrievers ───────────────────────────────────────────────────────
    logger.info("Loading dense retriever...")
    dense = DenseRetriever()
    dense.load_index()

    bm25 = None
    if args.retriever == "hybrid":
        logger.info("Loading BM25 retriever...")
        bm25 = BM25Retriever()
        bm25.load_bm25()

    # ── Categorize ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    n_eval = min(n_q, n_p)

    for i in range(n_q):
        refs = ref_answers.get(str(i), [])
        if not refs:
            logger.warning("No reference answer for question %d — skipping.", i)
            results.append({
                "idx": i, "question": questions[i], "references": [],
                "prediction": "", "em": 0.0, "f1": 0.0,
                "category": UNEVALUATED, "category_reason": "No reference answer found.",
            })
            continue

        if i >= n_p:
            results.append({
                "idx": i, "question": questions[i], "references": refs,
                "prediction": "", "em": 0.0, "f1": 0.0,
                "category": UNEVALUATED, "category_reason": "No prediction available (index out of range).",
            })
            continue

        pred = predictions[i]
        logger.info("[%d/%d] Categorizing: %s", i + 1, n_eval, questions[i][:60])

        result = categorize(
            idx=i,
            question=questions[i],
            references=refs,
            prediction=pred,
            norm_corpus=norm_corpus,
            chunks=chunks,
            dense=dense,
            bm25=bm25,
            top_k=args.top_k,
        )
        results.append(result)

    # ── LLM judge (optional) ───────────────────────────────────────────────────
    if args.llm_judge:
        run_llm_judge(results)

    # ── Output ─────────────────────────────────────────────────────────────────
    print(build_summary_table(results))

    if args.examples > 0:
        print_examples(results, n=args.examples)

    write_json_report(results, args.output_json)
    write_csv_report(results, args.output_csv)


if __name__ == "__main__":
    main()
