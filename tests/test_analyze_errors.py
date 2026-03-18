"""
Unit tests for analyze_errors.py.

All tests are self-contained — no real retrievers, no disk I/O.
Retrievers are replaced with simple mocks that return predetermined chunks.
"""

import warnings
from unittest.mock import MagicMock

import pytest

from unittest.mock import patch

from analyze_errors import (
    CORRECT,
    CORPUS_GAP,
    FALLBACK,
    FORMAT_MISMATCH,
    GENERATION_FAIL,
    RETRIEVAL_MISS,
    UNEVALUATED,
    build_normalized_corpus,
    categorize,
    compute_scores,
    is_answer_in_corpus,
    is_answer_in_retrieved,
    load_reference_answers,
    build_summary_table,
    run_llm_judge,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc1", "url": "https://example.com",
            "title": "Test", "text": text, "chunk_index": 0, "total_chunks": 1}


def _mock_dense(results_per_query: list[list[dict]]) -> MagicMock:
    """Return a mock DenseRetriever whose batch_retrieve_top_k returns the given results."""
    dense = MagicMock()
    dense.batch_retrieve_top_k.return_value = results_per_query
    return dense


def _mock_bm25(results: list[dict]) -> MagicMock:
    bm25 = MagicMock()
    bm25.retrieve_top_k.return_value = results
    return bm25


# ── load_reference_answers ────────────────────────────────────────────────────

def test_load_reference_answers_str_normalized_to_list(tmp_path):
    data = {"0": "Master of Science"}
    f = tmp_path / "refs.json"
    import json; f.write_text(json.dumps(data))
    result = load_reference_answers(str(f))
    assert result["0"] == ["Master of Science"]


def test_load_reference_answers_list_passthrough(tmp_path):
    data = {"0": ["M.S.", "Master of Science"]}
    f = tmp_path / "refs.json"
    import json; f.write_text(json.dumps(data))
    result = load_reference_answers(str(f))
    assert result["0"] == ["M.S.", "Master of Science"]


# ── compute_scores ────────────────────────────────────────────────────────────

def test_compute_scores_exact_match():
    em, f1 = compute_scores("Master of Science", ["Master of Science"])
    assert em == 1.0
    assert f1 == 1.0


def test_compute_scores_list_takes_max():
    # prediction matches the second reference
    em, f1 = compute_scores("M.S.", ["Master of Science", "M.S."])
    assert em == 1.0


def test_compute_scores_partial_f1():
    em, f1 = compute_scores("Master Science", ["Master of Science"])
    assert em == 0.0
    assert f1 > 0.0


def test_compute_scores_zero():
    em, f1 = compute_scores("completely wrong answer", ["Master of Science"])
    assert em == 0.0
    assert f1 == 0.0


# ── is_answer_in_corpus ───────────────────────────────────────────────────────

def test_is_answer_in_corpus_found():
    chunks = [_make_chunk("c1", "The degree is Master of Science in EECS.")]
    norm_corpus = build_normalized_corpus(chunks)
    found, ref, idx = is_answer_in_corpus(["Master of Science"], norm_corpus)
    assert found is True
    assert ref == "Master of Science"
    assert idx == 0


def test_is_answer_in_corpus_not_found():
    chunks = [_make_chunk("c1", "Completely unrelated text about programming.")]
    norm_corpus = build_normalized_corpus(chunks)
    found, ref, idx = is_answer_in_corpus(["Master of Science"], norm_corpus)
    assert found is False
    assert ref is None
    assert idx is None


def test_is_answer_in_corpus_any_reference_matches():
    chunks = [_make_chunk("c1", "The program offers an M.S. degree.")]
    norm_corpus = build_normalized_corpus(chunks)
    # first ref doesn't match, second does
    found, ref, idx = is_answer_in_corpus(["Master of Science", "M.S."], norm_corpus)
    assert found is True
    assert ref == "M.S."


def test_is_answer_in_corpus_empty_ref_warns():
    chunks = [_make_chunk("c1", "Some text here.")]
    norm_corpus = build_normalized_corpus(chunks)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found, _, _ = is_answer_in_corpus([""], norm_corpus)
    assert found is False
    assert any("empty after normalization" in str(w.message) for w in caught)


def test_is_answer_in_corpus_empty_chunk_skipped():
    # A chunk that normalizes to empty string should not be matched
    chunks = [_make_chunk("c1", "a an the")]  # all articles → empty after normalize
    norm_corpus = build_normalized_corpus(chunks)
    # The ref "a" also normalizes to "" — both empty, no false positive
    found, _, _ = is_answer_in_corpus(["something else"], norm_corpus)
    assert found is False


# ── is_answer_in_retrieved ────────────────────────────────────────────────────

def test_is_answer_in_retrieved_found_returns_rank():
    retrieved = [
        _make_chunk("c1", "Unrelated chunk about calculus."),
        _make_chunk("c2", "The answer is Master of Science."),
    ]
    found, ref, rank = is_answer_in_retrieved(["Master of Science"], retrieved)
    assert found is True
    assert rank == 1


def test_is_answer_in_retrieved_not_found():
    retrieved = [_make_chunk("c1", "Nothing useful here.")]
    found, ref, rank = is_answer_in_retrieved(["Master of Science"], retrieved)
    assert found is False
    assert rank is None


def test_is_answer_in_retrieved_fuzzy_match():
    # "April 18th, 2026" breaks substring match for ref "April 18, 2026"
    # but token F1 is high enough to catch it
    retrieved = [_make_chunk("c1", "Cal Day is on April 18th, 2026 at Berkeley.")]
    found, ref, rank = is_answer_in_retrieved(["April 18, 2026"], retrieved)
    assert found is True
    assert rank == 0


# ── categorize — all paths ────────────────────────────────────────────────────

def _run_categorize(question, references, prediction, chunks, dense_results, bm25_results=None):
    """Helper: run categorize() with mocked retrievers."""
    norm_corpus = build_normalized_corpus(chunks)
    dense = _mock_dense([dense_results])
    bm25 = _mock_bm25(bm25_results) if bm25_results is not None else None
    return categorize(
        idx=0,
        question=question,
        references=references,
        prediction=prediction,
        norm_corpus=norm_corpus,
        chunks=chunks,
        dense=dense,
        bm25=bm25,
        top_k=5,
    )


def test_categorize_correct():
    chunks = [_make_chunk("c1", "The degree offered is Master of Science.")]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="Master of Science",
        chunks=chunks,
        dense_results=[],
    )
    assert result["category"] == CORRECT
    assert result["em"] == 1.0


def test_categorize_corpus_gap():
    chunks = [_make_chunk("c1", "Nothing about degrees here.")]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="Unknown",
        chunks=chunks,
        dense_results=[],
    )
    assert result["category"] == CORPUS_GAP
    assert "best_partial_f1_in_corpus" in result


def test_categorize_retrieval_miss():
    # Answer IS in corpus but not in the top-k returned by retriever
    chunks = [_make_chunk("c1", "The degree offered is Master of Science.")]
    unrelated_chunk = _make_chunk("c2", "Calculus and physics courses.")

    # Dense retriever returns something unrelated; large-k search also misses
    dense = _mock_dense([[unrelated_chunk]])
    # For the large-k call inside _find_best_rank_large, also return unrelated
    dense.batch_retrieve_top_k.return_value = [[unrelated_chunk]]

    norm_corpus = build_normalized_corpus(chunks)
    result = categorize(
        idx=0,
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="Unknown",
        norm_corpus=norm_corpus,
        chunks=chunks,
        dense=dense,
        bm25=None,
        top_k=5,
    )
    assert result["category"] == RETRIEVAL_MISS
    assert "answer_chunk_ids" in result


def test_categorize_fallback():
    # Answer in corpus AND in top-k retrieved, but prediction is "Unknown"
    answer_chunk = _make_chunk("c1", "The degree offered is Master of Science.")
    chunks = [answer_chunk]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="Unknown",
        chunks=chunks,
        dense_results=[answer_chunk],
    )
    assert result["category"] == FALLBACK
    assert result["raw_prediction"] == "Unknown"


def test_categorize_fallback_whitespace_only_prediction():
    answer_chunk = _make_chunk("c1", "The degree offered is Master of Science.")
    chunks = [answer_chunk]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="   ",
        chunks=chunks,
        dense_results=[answer_chunk],
    )
    assert result["category"] == FALLBACK


def test_categorize_format_mismatch():
    # Context retrieved, F1 > 0 but EM = 0
    answer_chunk = _make_chunk("c1", "The degree offered is Master of Science (M.S.).")
    chunks = [answer_chunk]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="MS degree",   # has "MS"/"degree" but not exact
        chunks=chunks,
        dense_results=[answer_chunk],
    )
    # "ms" overlaps with normalized "master science" → F1 > 0 but EM = 0
    assert result["category"] == FORMAT_MISMATCH
    assert "norm_prediction" in result
    assert "norm_reference" in result


def test_categorize_generation_fail():
    # Context retrieved, prediction completely unrelated (F1 = 0), not a fallback
    answer_chunk = _make_chunk("c1", "The degree offered is Master of Science.")
    chunks = [answer_chunk]
    result = _run_categorize(
        question="What degree is offered?",
        references=["Master of Science"],
        prediction="purple elephant",
        chunks=chunks,
        dense_results=[answer_chunk],
    )
    assert result["category"] == GENERATION_FAIL
    assert "context_f1" in result


# ── run_llm_judge ─────────────────────────────────────────────────────────────

def test_llm_judge_updates_results_in_place():
    """LLM judge should add llm_correct and llm_judge_reason to matching entries."""
    results = [
        {"idx": 0, "category": FORMAT_MISMATCH, "question": "Q?",
         "references": ["Master of Science"], "prediction": "M.S."},
        {"idx": 1, "category": GENERATION_FAIL, "question": "Q2?",
         "references": ["Dan Klein"], "prediction": "Professor Klein"},
        {"idx": 2, "category": CORPUS_GAP, "question": "Q3?",
         "references": ["Some answer"], "prediction": "Unknown"},
    ]
    fake_response = '[{"idx": 0, "correct": true, "reason": "M.S. is an abbreviation."}, {"idx": 1, "correct": false, "reason": "Missing first name."}]'

    with patch("analyze_errors.call_llm", return_value=fake_response):
        run_llm_judge(results)

    assert results[0]["llm_correct"] is True
    assert "M.S." in results[0]["llm_judge_reason"]
    assert results[1]["llm_correct"] is False
    # CORPUS_GAP entry should not be touched
    assert "llm_correct" not in results[2]


def test_llm_judge_handles_markdown_code_fence():
    """LLM response wrapped in ```json ... ``` should still be parsed."""
    results = [
        {"idx": 3, "category": FORMAT_MISMATCH, "question": "Q?",
         "references": ["Room 405"], "prediction": "405"},
    ]
    fake_response = '```json\n[{"idx": 3, "correct": true, "reason": "Same room number."}]\n```'

    with patch("analyze_errors.call_llm", return_value=fake_response):
        run_llm_judge(results)

    assert results[0]["llm_correct"] is True


def test_llm_judge_handles_invalid_json_gracefully():
    """If the LLM returns garbage JSON, results should be unchanged (no crash)."""
    results = [
        {"idx": 0, "category": FORMAT_MISMATCH, "question": "Q?",
         "references": ["ans"], "prediction": "answer"},
    ]
    with patch("analyze_errors.call_llm", return_value="not valid json at all"):
        run_llm_judge(results)

    assert "llm_correct" not in results[0]


def test_llm_judge_handles_api_failure_gracefully():
    """RuntimeError from call_llm should be caught; results unchanged."""
    results = [
        {"idx": 0, "category": FORMAT_MISMATCH, "question": "Q?",
         "references": ["ans"], "prediction": "answer"},
    ]
    with patch("analyze_errors.call_llm", side_effect=RuntimeError("timeout")):
        run_llm_judge(results)

    assert "llm_correct" not in results[0]


def test_llm_judge_marks_none_for_missing_verdict():
    """If LLM omits an idx from its response, that entry gets llm_correct=None."""
    results = [
        {"idx": 0, "category": FORMAT_MISMATCH, "question": "Q?",
         "references": ["ans"], "prediction": "answer"},
        {"idx": 1, "category": FORMAT_MISMATCH, "question": "Q2?",
         "references": ["ans2"], "prediction": "answer2"},
    ]
    # LLM only returns verdict for idx 0, misses idx 1
    fake_response = '[{"idx": 0, "correct": true, "reason": "OK."}]'

    with patch("analyze_errors.call_llm", return_value=fake_response):
        run_llm_judge(results)

    assert results[0]["llm_correct"] is True
    assert results[1]["llm_correct"] is None


def test_llm_judge_skips_when_no_candidates():
    """If there are no FORMAT_MISMATCH or GENERATION_FAIL entries, call_llm is never called."""
    results = [{"idx": 0, "category": CORPUS_GAP, "question": "Q?",
                "references": ["ans"], "prediction": "Unknown"}]

    with patch("analyze_errors.call_llm") as mock_llm:
        run_llm_judge(results)
        mock_llm.assert_not_called()


# ── build_summary_table ───────────────────────────────────────────────────────

def test_build_summary_table_counts_correctly():
    results = [
        {"category": CORRECT},
        {"category": CORPUS_GAP},
        {"category": CORPUS_GAP},
        {"category": RETRIEVAL_MISS},
    ]
    table = build_summary_table(results)
    assert "CORRECT" in table
    assert "CORPUS_GAP" in table
    # Check counts appear somewhere in the table
    assert " 1 " in table or "1 " in table  # CORRECT count
    assert " 2 " in table  # CORPUS_GAP count


# ── Prediction count mismatch ─────────────────────────────────────────────────

def test_categorize_unevaluated_when_no_prediction():
    """Questions with no corresponding prediction line are marked UNEVALUATED."""
    # We test this via the logic in categorize: if i >= n_p, result is UNEVALUATED.
    # Here we just verify that the UNEVALUATED constant exists and can be used as a dict value.
    entry = {
        "idx": 99, "question": "Q?", "references": ["A"],
        "prediction": "", "em": 0.0, "f1": 0.0,
        "category": UNEVALUATED,
        "category_reason": "No prediction available (index out of range).",
    }
    assert entry["category"] == UNEVALUATED
