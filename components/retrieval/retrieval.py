"""
Retrieval Component
Handles semantic search, BM25, hybrid retrieval, MMR, and optional reranking.
"""

import os
import json
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Retrieval Strategies
# ─────────────────────────────────────────────

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass


class SemanticRetriever(BaseRetriever):
    """Pure dense vector similarity search."""

    def __init__(self, embedding_component, score_threshold: float = 0.0):
        self.embedding_component = embedding_component
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.embedding_component.search(query, top_k=top_k)
        return [r for r in results if r["score"] >= self.score_threshold]


class BM25Retriever(BaseRetriever):
    """Sparse BM25 keyword retrieval."""

    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        word_doc_freq = defaultdict(int)

        for doc in self.corpus:
            tokens = self._tokenize(doc["content"])
            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self.doc_freqs.append(freq)
            for word in set(tokens):
                word_doc_freq[word] += 1

        self.avgdl = sum(sum(f.values()) for f in self.doc_freqs) / max(1, len(self.corpus))

        N = len(self.corpus)
        for word, df in word_doc_freq.items():
            self.idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = sum(doc_freq.values())
        for token in query_tokens:
            if token not in doc_freq:
                continue
            idf = self.idf.get(token, 0)
            tf = doc_freq[token]
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scores = [(i, self._score(query_tokens, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                doc = self.corpus[idx].copy()
                doc["score"] = score
                results.append(doc)
        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval: dense + sparse with Reciprocal Rank Fusion."""

    def __init__(self, semantic: SemanticRetriever, bm25: BM25Retriever,
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.semantic = semantic
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]], k: int = 60) -> List[Tuple[str, float]]:
        """RRF score fusion across multiple ranked lists."""
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get("chunk_id") or doc.get("content", "")[:50]
                scores[doc_id] += 1 / (k + rank + 1)
                doc_map[doc_id] = doc

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in sorted_docs], doc_map

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fetch_k = top_k * 3
        dense_results = self.semantic.retrieve(query, top_k=fetch_k)
        sparse_results = self.bm25.retrieve(query, top_k=fetch_k)

        sorted_docs, doc_map = self._reciprocal_rank_fusion([dense_results, sparse_results])
        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = doc_map[doc_id].copy()
            doc["score"] = rrf_score
            doc["retrieval_method"] = "hybrid"
            results.append(doc)
        return results


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance - balances relevance and diversity."""

    def __init__(self, embedding_component, fetch_k: int = 20, lambda_mult: float = 0.5):
        self.embedding_component = embedding_component
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Get more candidates than needed
        candidates = self.embedding_component.search(query, top_k=self.fetch_k)
        if not candidates:
            return []

        query_embedding = self.embedding_component.embed_query(query)
        selected = []
        selected_embeddings = []
        remaining = list(enumerate(candidates))

        while len(selected) < top_k and remaining:
            best_idx = None
            best_score = -1

            for i, doc in remaining:
                # We'd need doc embeddings here - simplified: use score as relevance
                relevance = doc["score"]
                if not selected_embeddings:
                    mmr_score = relevance
                else:
                    # Approximate diversity using content similarity
                    diversity = min(
                        1 - self._cosine_similarity(
                            [hash(c) % 1 for c in doc["content"][:100]],
                            [hash(c) % 1 for c in s["content"][:100]]
                        )
                        for s in selected
                    )
                    mmr_score = self.lambda_mult * relevance + (1 - self.lambda_mult) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected.append(candidates[best_idx])
                remaining = [(i, d) for i, d in remaining if i != best_idx]

        return selected


# ─────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────

class CohereReranker:
    def __init__(self, config: Dict[str, Any]):
        import cohere
        self.client = cohere.Client(config.get("api_key") or os.environ["COHERE_API_KEY"])
        self.model = config.get("model", "rerank-english-v3.0")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        texts = [d["content"] for d in documents]
        response = self.client.rerank(query=query, documents=texts, model=self.model, top_n=top_k)
        reranked = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc["score"] = result.relevance_score
            doc["reranked"] = True
            reranked.append(doc)
        return reranked


class CrossEncoderReranker:
    def __init__(self, config: Dict[str, Any]):
        from sentence_transformers import CrossEncoder
        model_name = config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        pairs = [(query, d["content"]) for d in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in ranked[:top_k]:
            d = doc.copy()
            d["score"] = float(score)
            d["reranked"] = True
            results.append(d)
        return results


# ─────────────────────────────────────────────
# Retrieval Component
# ─────────────────────────────────────────────

class RetrievalComponent:
    def __init__(self, config: Dict[str, Any], embedding_component=None, corpus: Optional[List[Dict]] = None):
        self.config = config
        self.strategy = config.get("strategy", "semantic")
        self.top_k = config.get("top_k", 5)
        self.score_threshold = config.get("score_threshold", 0.0)
        self.embedding_component = embedding_component
        self.corpus = corpus or []
        self._build_retriever()
        self._build_reranker()

    def _build_retriever(self):
        if self.strategy == "semantic":
            self.retriever = SemanticRetriever(self.embedding_component, self.score_threshold)

        elif self.strategy == "bm25":
            self.retriever = BM25Retriever(self.corpus)

        elif self.strategy == "hybrid":
            hybrid_cfg = self.config.get("hybrid", {})
            semantic = SemanticRetriever(self.embedding_component, self.score_threshold)
            bm25 = BM25Retriever(self.corpus)
            self.retriever = HybridRetriever(
                semantic, bm25,
                dense_weight=hybrid_cfg.get("dense_weight", 0.7),
                sparse_weight=hybrid_cfg.get("sparse_weight", 0.3),
            )

        elif self.strategy == "mmr":
            mmr_cfg = self.config.get("mmr", {})
            self.retriever = MMRRetriever(
                self.embedding_component,
                fetch_k=mmr_cfg.get("fetch_k", 20),
                lambda_mult=mmr_cfg.get("lambda_mult", 0.5),
            )
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.strategy}")

    def _build_reranker(self):
        reranker_cfg = self.config.get("reranker", {})
        if not reranker_cfg.get("enabled", False):
            self.reranker = None
            return
        provider = reranker_cfg.get("provider", "cohere")
        if provider == "cohere":
            self.reranker = CohereReranker(reranker_cfg)
        elif provider == "cross-encoder":
            self.reranker = CrossEncoderReranker(reranker_cfg)
        else:
            self.reranker = None

    def retrieve(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        logger.info(f"Retrieving top-{k} docs for query: '{query[:80]}...'")

        results = self.retriever.retrieve(query, top_k=k)

        if self.reranker:
            logger.info(f"Reranking {len(results)} results...")
            results = self.reranker.rerank(query, results, top_k=k)

        logger.info(f"Retrieved {len(results)} documents")
        return results

    def retrieve_and_save(self, query: str, output_path: str, top_k: Optional[int] = None) -> str:
        results = self.retrieve(query, top_k=top_k)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"query": query, "results": results}, f, indent=2)
        return output_path


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="RAG Retrieval Component")
    parser.add_argument("--query", required=True)
    parser.add_argument("--chunks-path", help="Path to corpus chunks JSON (for BM25/hybrid)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    corpus = []
    if args.chunks_path:
        with open(args.chunks_path) as f:
            corpus = json.load(f)

    from components.embedding.embedding import EmbeddingComponent
    emb_component = EmbeddingComponent(cfg.get("embedding", {}), cfg.get("vector_store", {}))

    component = RetrievalComponent(cfg.get("retrieval", {}), embedding_component=emb_component, corpus=corpus)
    component.retrieve_and_save(args.query, args.output_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
