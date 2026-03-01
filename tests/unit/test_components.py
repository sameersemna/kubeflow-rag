"""
Unit Tests for Kubeflow RAG Template Components
Run with: pytest tests/unit/ -v
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add root to path
sys.path.insert(0, str(Path(__file__).parents[2]))


# ─────────────────────────────────────────────
# Ingestion Tests
# ─────────────────────────────────────────────

class TestTextSplitter:
    def setup_method(self):
        from components.ingestion.ingestion import TextSplitter
        self.splitter = TextSplitter(chunk_size=100, chunk_overlap=20, strategy="recursive")

    def test_short_text_no_split(self):
        text = "Hello world"
        chunks = self.splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_splits(self):
        text = "word " * 50  # 250 chars
        chunks = self.splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100 + 20  # Allow slight overlap

    def test_character_strategy(self):
        from components.ingestion.ingestion import TextSplitter
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10, strategy="character")
        text = "a" * 200
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = self.splitter.split("")
        assert chunks == [] or all(c.strip() for c in chunks)


class TestIngestionComponent:
    def setup_method(self):
        from components.ingestion.ingestion import IngestionComponent
        self.component = IngestionComponent({"chunk_size": 200, "chunk_overlap": 20})

    def test_ingest_txt_file(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("Hello world! " * 50)

        chunks = self.component.ingest([str(doc)])
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.source == str(doc)

    def test_ingest_json_file(self, tmp_path):
        doc = tmp_path / "test.json"
        doc.write_text(json.dumps([{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]))

        chunks = self.component.ingest([str(doc)])
        assert len(chunks) > 0

    def test_ingest_markdown_file(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("# Title\n\nSome content " * 30)

        chunks = self.component.ingest([str(doc)])
        assert len(chunks) > 0

    def test_ingest_to_json(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("Test content " * 30)
        output = str(tmp_path / "chunks.json")

        self.component.ingest_to_json([str(doc)], output)
        assert Path(output).exists()
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "chunk_id" in data[0]
        assert "content" in data[0]

    def test_chunk_id_uniqueness(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("Different content " * 100)

        chunks = self.component.ingest([str(doc)])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_invalid_source_graceful_failure(self):
        chunks = self.component.ingest(["/nonexistent/file.pdf"])
        assert chunks == []


# ─────────────────────────────────────────────
# Retrieval Tests
# ─────────────────────────────────────────────

class TestBM25Retriever:
    def setup_method(self):
        from components.retrieval.retrieval import BM25Retriever
        self.corpus = [
            {"chunk_id": "1", "content": "Python is a great programming language for data science"},
            {"chunk_id": "2", "content": "Machine learning requires large amounts of training data"},
            {"chunk_id": "3", "content": "Kubernetes orchestrates containerized applications"},
            {"chunk_id": "4", "content": "Docker containers package applications and dependencies"},
            {"chunk_id": "5", "content": "Python machine learning libraries include scikit-learn and tensorflow"},
        ]
        self.retriever = BM25Retriever(self.corpus)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("Python programming", top_k=3)
        assert len(results) > 0

    def test_retrieve_relevance_ordering(self):
        results = self.retriever.retrieve("Python machine learning", top_k=5)
        # Python/ML documents should rank higher
        assert any("Python" in r["content"] or "machine learning" in r["content"] for r in results[:2])

    def test_retrieve_top_k(self):
        results = self.retriever.retrieve("technology", top_k=2)
        assert len(results) <= 2

    def test_retrieve_no_match(self):
        results = self.retriever.retrieve("cooking recipes pasta", top_k=3)
        # Should return results but with low scores
        # (BM25 may still return something)
        assert isinstance(results, list)

    def test_score_ordering(self):
        results = self.retriever.retrieve("Python", top_k=5)
        if len(results) >= 2:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]


class TestContextPrecision:
    def setup_method(self):
        from components.evaluation.evaluation import ContextPrecision
        self.metric = ContextPrecision()

    def test_perfect_relevance(self):
        score = self.metric.score(
            "What is Python?",
            ["Python is a programming language"],
            "Python is used for programming",
        )
        assert 0 <= score <= 1

    def test_empty_contexts(self):
        score = self.metric.score("Question?", [], "Answer")
        assert score == 0.0


# ─────────────────────────────────────────────
# Evaluation Tests
# ─────────────────────────────────────────────

class TestEvaluationComponent:
    def setup_method(self):
        from components.evaluation.evaluation import EvaluationComponent
        self.component = EvaluationComponent({
            "framework": "custom",
            "thresholds": {"keyword_faithfulness": 0.3, "answer_relevancy": 0.3},
        })

    def test_evaluate_returns_report(self):
        retrieved = [{"content": "Python is a programming language", "score": 0.9, "metadata": {}}]
        report = self.component.evaluate("What is Python?", "Python is a programming language.", retrieved)
        assert report.query == "What is Python?"
        assert 0 <= report.overall_score <= 1
        assert isinstance(report.passed, bool)
        assert len(report.results) > 0

    def test_evaluate_batch(self):
        test_cases = [
            {
                "question": "What is Docker?",
                "answer": "Docker is a container platform.",
                "retrieved_docs": [{"content": "Docker containers", "score": 0.8, "metadata": {}}],
            },
            {
                "question": "What is Kubernetes?",
                "answer": "Kubernetes orchestrates containers.",
                "retrieved_docs": [{"content": "Kubernetes orchestration", "score": 0.9, "metadata": {}}],
            },
        ]
        result = self.component.evaluate_batch(test_cases)
        assert result["total_cases"] == 2
        assert "pass_rate" in result
        assert "average_scores" in result

    def test_process_from_files(self, tmp_path):
        gen_output = tmp_path / "generation.json"
        ret_output = tmp_path / "retrieval.json"
        eval_output = tmp_path / "evaluation.json"

        gen_output.write_text(json.dumps({"query": "Test?", "answer": "Test answer."}))
        ret_output.write_text(json.dumps({"query": "Test?", "results": [{"content": "Test context", "score": 0.8, "metadata": {}}]}))

        self.component.process_from_files(str(gen_output), str(ret_output), str(eval_output))
        assert eval_output.exists()

        with open(eval_output) as f:
            data = json.load(f)
        assert "overall_score" in data


# ─────────────────────────────────────────────
# Integration-like tests (no external services)
# ─────────────────────────────────────────────

class TestEndToEndLocal:
    """End-to-end test using only local/in-memory components."""

    def test_ingest_then_bm25_retrieve(self, tmp_path):
        from components.ingestion.ingestion import IngestionComponent
        from components.retrieval.retrieval import BM25Retriever

        # Create doc
        doc = tmp_path / "test.txt"
        doc.write_text("Machine learning is the future of AI. " * 20 + "Docker makes deployment easy. " * 20)

        # Ingest
        ingestion = IngestionComponent({"chunk_size": 200, "chunk_overlap": 20})
        chunks_path = str(tmp_path / "chunks.json")
        ingestion.ingest_to_json([str(doc)], chunks_path)

        with open(chunks_path) as f:
            chunks = json.load(f)

        # BM25 retrieve
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("machine learning AI", top_k=3)

        assert len(results) > 0
        assert any("machine learning" in r["content"].lower() or "ai" in r["content"].lower() for r in results)

    def test_full_pipeline_with_evaluation(self, tmp_path):
        from components.ingestion.ingestion import IngestionComponent
        from components.retrieval.retrieval import BM25Retriever
        from components.evaluation.evaluation import EvaluationComponent

        # Create docs
        doc = tmp_path / "knowledge.txt"
        doc.write_text(
            "The capital of France is Paris. Paris is a major European city. " * 15 +
            "Python was created by Guido van Rossum in 1991. Python is widely used. " * 15
        )

        # Ingest
        ingestion = IngestionComponent({"chunk_size": 150, "chunk_overlap": 30})
        chunks_path = str(tmp_path / "chunks.json")
        ingestion.ingest_to_json([str(doc)], chunks_path)

        with open(chunks_path) as f:
            chunks = json.load(f)

        # Retrieve
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("What is the capital of France?", top_k=3)

        # Evaluate
        evaluator = EvaluationComponent({"framework": "custom", "thresholds": {"keyword_faithfulness": 0.1}})
        report = evaluator.evaluate(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            retrieved_docs=results,
        )

        assert report.overall_score > 0
        assert len(report.results) > 0
