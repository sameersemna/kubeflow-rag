"""
Evaluation Component
Evaluates RAG pipeline quality using RAGAS, TruLens, or custom metrics.
Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    metric: str
    score: float
    passed: bool
    threshold: float
    details: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineEvaluationReport:
    query: str
    answer: str
    results: List[EvaluationResult]
    overall_score: float
    passed: bool

    def to_dict(self):
        d = asdict(self)
        d["results"] = [r for r in d["results"]]
        return d


# ─────────────────────────────────────────────
# Custom Metrics (no external dependency)
# ─────────────────────────────────────────────

class KeywordFaithfulness:
    """Simple keyword overlap faithfulness check."""
    name = "keyword_faithfulness"

    def score(self, answer: str, contexts: List[str]) -> float:
        import re
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        context_text = " ".join(contexts)
        context_words = set(re.findall(r'\b\w+\b', context_text.lower()))
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had",
                     "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
                     "of", "in", "on", "at", "to", "for", "with", "by", "from", "up", "about", "than",
                     "this", "that", "these", "those", "and", "or", "but", "if", "then", "so"}
        answer_words -= stopwords
        if not answer_words:
            return 1.0
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)


class SemanticAnswerRelevancy:
    """Semantic similarity between question and answer."""
    name = "answer_relevancy"

    def __init__(self, embedding_fn=None):
        self.embedding_fn = embedding_fn

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def score(self, question: str, answer: str) -> float:
        if self.embedding_fn is None:
            # Fallback: word overlap
            import re
            q_words = set(re.findall(r'\b\w+\b', question.lower()))
            a_words = set(re.findall(r'\b\w+\b', answer.lower()))
            if not q_words:
                return 0.0
            return len(q_words & a_words) / len(q_words)
        q_emb = self.embedding_fn(question)
        a_emb = self.embedding_fn(answer)
        return self._cosine_sim(q_emb, a_emb)


class ContextPrecision:
    """What fraction of retrieved context is actually relevant."""
    name = "context_precision"

    def score(self, question: str, contexts: List[str], answer: str) -> float:
        import re
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        a_words = set(re.findall(r'\b\w+\b', answer.lower()))
        relevant_words = q_words | a_words
        if not relevant_words:
            return 1.0
        relevant_ctx = 0
        for ctx in contexts:
            ctx_words = set(re.findall(r'\b\w+\b', ctx.lower()))
            if len(ctx_words & relevant_words) / len(relevant_words) > 0.2:
                relevant_ctx += 1
        return relevant_ctx / len(contexts) if contexts else 0.0


# ─────────────────────────────────────────────
# RAGAS Evaluator
# ─────────────────────────────────────────────

class RagasEvaluator:
    """Uses RAGAS library for evaluation."""

    def __init__(self, config: Dict[str, Any]):
        self.metrics_to_run = config.get("metrics", ["faithfulness", "answer_relevancy"])
        self.llm_provider = config.get("llm_provider", "openai")
        self._setup()

    def _setup(self):
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            self.evaluate = evaluate
            self.metrics_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            self.available = True
        except ImportError:
            logger.warning("RAGAS not installed. Falling back to custom metrics. Install with: pip install ragas")
            self.available = False

    def evaluate_single(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
        if not self.available:
            raise RuntimeError("RAGAS not installed")

        from datasets import Dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics = [self.metrics_map[m] for m in self.metrics_to_run if m in self.metrics_map]
        result = self.evaluate(dataset, metrics=metrics)
        return dict(result)


# ─────────────────────────────────────────────
# Evaluation Component
# ─────────────────────────────────────────────

class EvaluationComponent:
    def __init__(self, config: Dict[str, Any], embedding_fn=None):
        self.config = config
        self.framework = config.get("framework", "custom")
        self.thresholds = config.get("thresholds", {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
        })
        self.embedding_fn = embedding_fn
        self._setup_metrics()

    def _setup_metrics(self):
        self.custom_metrics = {
            "keyword_faithfulness": KeywordFaithfulness(),
            "answer_relevancy": SemanticAnswerRelevancy(self.embedding_fn),
            "context_precision": ContextPrecision(),
        }
        if self.framework == "ragas":
            try:
                self.ragas = RagasEvaluator(self.config)
            except Exception as e:
                logger.warning(f"RAGAS setup failed: {e}. Using custom metrics.")
                self.framework = "custom"
                self.ragas = None
        else:
            self.ragas = None

    def _run_custom_metrics(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        scores = {}
        scores["keyword_faithfulness"] = self.custom_metrics["keyword_faithfulness"].score(answer, contexts)
        scores["answer_relevancy"] = self.custom_metrics["answer_relevancy"].score(question, answer)
        scores["context_precision"] = self.custom_metrics["context_precision"].score(question, contexts, answer)
        return scores

    def evaluate(self, question: str, answer: str, retrieved_docs: List[Dict[str, Any]],
                 ground_truth: Optional[str] = None) -> PipelineEvaluationReport:
        contexts = [d["content"] for d in retrieved_docs]

        if self.framework == "ragas" and self.ragas and self.ragas.available:
            try:
                raw_scores = self.ragas.evaluate_single(question, answer, contexts, ground_truth)
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}. Using custom metrics.")
                raw_scores = self._run_custom_metrics(question, answer, contexts)
        else:
            raw_scores = self._run_custom_metrics(question, answer, contexts)

        results = []
        for metric, score in raw_scores.items():
            threshold = self.thresholds.get(metric, 0.5)
            results.append(EvaluationResult(
                metric=metric,
                score=round(float(score), 4),
                passed=float(score) >= threshold,
                threshold=threshold,
            ))

        overall_score = sum(r.score for r in results) / len(results) if results else 0.0
        passed = all(r.passed for r in results)

        report = PipelineEvaluationReport(
            query=question,
            answer=answer[:500],
            results=results,
            overall_score=round(overall_score, 4),
            passed=passed,
        )
        logger.info(f"Evaluation: overall={overall_score:.3f}, passed={passed}")
        return report

    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of test cases."""
        reports = []
        for case in test_cases:
            report = self.evaluate(
                question=case["question"],
                answer=case["answer"],
                retrieved_docs=case.get("retrieved_docs", []),
                ground_truth=case.get("ground_truth"),
            )
            reports.append(report.to_dict())

        # Aggregate
        all_scores: Dict[str, List[float]] = {}
        for report in reports:
            for result in report["results"]:
                m = result["metric"]
                all_scores.setdefault(m, []).append(result["score"])

        avg_scores = {m: round(sum(s) / len(s), 4) for m, s in all_scores.items()}
        overall_pass_rate = sum(r["passed"] for r in reports) / len(reports) if reports else 0

        return {
            "total_cases": len(reports),
            "pass_rate": round(overall_pass_rate, 4),
            "average_scores": avg_scores,
            "reports": reports,
        }

    def process_from_files(self, generation_output_path: str, retrieval_output_path: str,
                           output_path: str, ground_truth_path: Optional[str] = None) -> str:
        with open(generation_output_path) as f:
            gen_data = json.load(f)
        with open(retrieval_output_path) as f:
            ret_data = json.load(f)

        ground_truth = None
        if ground_truth_path and Path(ground_truth_path).exists():
            with open(ground_truth_path) as f:
                gt_data = json.load(f)
            ground_truth = gt_data.get("ground_truth")

        report = self.evaluate(
            question=gen_data["query"],
            answer=gen_data["answer"],
            retrieved_docs=ret_data.get("results", []),
            ground_truth=ground_truth,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")
        return output_path


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="RAG Evaluation Component")
    parser.add_argument("--generation-output", required=True)
    parser.add_argument("--retrieval-output", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    component = EvaluationComponent(cfg.get("evaluation", {}))
    component.process_from_files(
        args.generation_output,
        args.retrieval_output,
        args.output_path,
        args.ground_truth,
    )


if __name__ == "__main__":
    main()
