"""
Prompt iteration dry-run: renders the exact system + user messages
the LLM will see for a few sample questions with mock passages.
No API call — just prints to stdout for eyeballing.

Usage:
    python3 prompt_dryrun.py
"""

import json
import random

from llms.llm_pipeline import SYSTEM_PROMPT, build_query

QUESTIONS_PATH = "questions.txt"
REFERENCE_ANSWERS_PATH = "reference_answers.json"
CHUNKS_JSONL_PATH = "data/chunks.jsonl"

# How many questions to preview
NUM_SAMPLES = 5
# How many passages per question (simulates top-k retrieval)
PASSAGES_PER_QUESTION = 5

SEPARATOR = "=" * 80


def load_chunks(path: str, limit: int = 500) -> list[dict]:
    """Load first `limit` chunks from JSONL for mock passages."""
    chunks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def main():
    with open(QUESTIONS_PATH) as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(REFERENCE_ANSWERS_PATH) as f:
        ref_answers = json.load(f)

    chunks = load_chunks(CHUNKS_JSONL_PATH)

    # Pick sample questions spread across the set
    indices = [0, len(questions) // 4, len(questions) // 2,
               3 * len(questions) // 4, len(questions) - 1]
    indices = indices[:NUM_SAMPLES]

    random.seed(42)

    for idx in indices:
        question = questions[idx]
        reference = ref_answers[str(idx)]

        # Pick random passages as mock retrieval results
        mock_passages = random.sample(chunks, min(PASSAGES_PER_QUESTION, len(chunks)))
        passages = [{"title": c.get("title", ""), "text": c["text"]} for c in mock_passages]

        user_message = build_query(question, passages)

        print(SEPARATOR)
        print(f"QUESTION #{idx}: {question}")
        print(f"EXPECTED ANSWER: {reference}")
        print(SEPARATOR)
        print()
        print(">>> SYSTEM PROMPT:")
        print(SYSTEM_PROMPT)
        print()
        print(">>> USER MESSAGE:")
        print(user_message)
        print()
        print(f">>> Token estimate: ~{len(user_message.split())} words")
        print(f">>> Passage count: {len(passages)}")
        print()
        print()


if __name__ == "__main__":
    main()
