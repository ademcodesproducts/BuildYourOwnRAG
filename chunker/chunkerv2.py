"""
Chunker v2: splits corpus documents into fixed-size word chunks with overlap.

Short documents (fewer words than CHUNK_MIN_DOC_WORDS) are emitted as a
single chunk.  Longer documents are split into fixed-size windows of
CHUNK_SIZE_WORDS words with CHUNK_OVERLAP_WORDS overlap.
"""

import json
import logging
import os

import config

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
    min_doc_words: int = config.CHUNK_MIN_DOC_WORDS,
) -> list[str]:

    words = text.split()

    if len(words) <= min_doc_words:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step

    return chunks


def chunk_corpus(
    input_path: str = config.CORPUS_COMBINED_PATH,
    output_path: str = config.CHUNKS_JSONL_PATH,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
    min_doc_words: int = config.CHUNK_MIN_DOC_WORDS,
) -> tuple[int, int]:

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_chunks = 0
    total_docs = 0

    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            doc = json.loads(line)
            total_docs += 1

            chunks = chunk_text(
                doc["text"],
                chunk_size=chunk_size,
                overlap=overlap,
                min_doc_words=min_doc_words,
            )

            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc['id']}_{i}",
                    "doc_id": doc["id"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info(
        "Chunking complete: %d docs → %d chunks → %s",
        total_docs,
        total_chunks,
        output_path,
    )
    return total_chunks, total_docs
