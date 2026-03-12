"""Tests for chunker.chunker.chunk_text."""

import json
import os
import tempfile

from chunker.chunker import chunk_text, chunk_corpus


class TestChunkText:
    def test_short_doc_kept_whole(self):
        """Documents under min_doc_words are returned as a single chunk."""
        text = "word " * 50  # 50 words
        chunks = chunk_text(text.strip(), chunk_size=200, overlap=50, min_doc_words=200)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_exact_boundary(self):
        """A doc with exactly min_doc_words words stays whole."""
        text = " ".join(f"w{i}" for i in range(200))
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_doc_words=200)
        assert len(chunks) == 1

    def test_just_over_boundary(self):
        """A doc with min_doc_words + 1 words gets chunked."""
        text = " ".join(f"w{i}" for i in range(201))
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_doc_words=200)
        assert len(chunks) == 2

    def test_chunk_sizes(self):
        """Each chunk (except possibly the last) has chunk_size words."""
        text = " ".join(f"w{i}" for i in range(500))
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_doc_words=200)
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 200
        # Last chunk can be shorter
        assert len(chunks[-1].split()) <= 200

    def test_overlap_content(self):
        """The overlap region between consecutive chunks has the right words."""
        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_doc_words=200)

        # Second chunk should start with words[150] (step = 200 - 50 = 150)
        second_words = chunks[1].split()
        assert second_words[0] == "w150"

        # First 50 words of chunk 2 should equal last 50 words of chunk 1
        first_tail = chunks[0].split()[-50:]
        second_head = chunks[1].split()[:50]
        assert first_tail == second_head

    def test_no_words_lost(self):
        """All original words appear in at least one chunk."""
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_doc_words=200)

        all_chunk_words: set[str] = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        assert all_chunk_words == set(words)

    def test_no_overlap_mode(self):
        """With overlap=0, chunks are non-overlapping."""
        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=0, min_doc_words=200)
        assert len(chunks) == 2
        assert chunks[0].split()[-1] == "w199"
        assert chunks[1].split()[0] == "w200"


class TestChunkCorpus:
    def test_round_trip(self, tmp_path):
        """chunk_corpus reads JSONL and writes chunked JSONL."""
        input_path = str(tmp_path / "corpus.jsonl")
        output_path = str(tmp_path / "chunks.jsonl")

        # Write a small corpus: one short doc, one long doc
        docs = [
            {"id": "aaa", "url": "http://a.com", "title": "Short", "text": "hello world"},
            {"id": "bbb", "url": "http://b.com", "title": "Long",
             "text": " ".join(f"w{i}" for i in range(500))},
        ]
        with open(input_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        total_chunks, total_docs = chunk_corpus(
            input_path=input_path,
            output_path=output_path,
            chunk_size=200, overlap=50, min_doc_words=200,
        )

        assert total_docs == 2
        # Short doc → 1 chunk, long doc (500 words) → multiple chunks
        assert total_chunks > 2

        # Verify output schema
        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == total_chunks
        for rec in records:
            assert "chunk_id" in rec
            assert "doc_id" in rec
            assert "text" in rec
            assert "chunk_index" in rec
            assert "total_chunks" in rec

        # First record should be the short doc, kept whole
        assert records[0]["doc_id"] == "aaa"
        assert records[0]["chunk_index"] == 0
        assert records[0]["total_chunks"] == 1
