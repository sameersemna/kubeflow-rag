"""
Embedding Component
Generates vector embeddings and stores them in the configured vector store.
Supports: OpenAI, Anthropic, HuggingFace, Cohere, Ollama
Vector Stores: Chroma, Weaviate, Qdrant, Pinecone, Milvus, PGVector
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Embedding Providers
# ─────────────────────────────────────────────

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class OpenAIEmbedding(BaseEmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.get("api_key") or os.environ["OPENAI_API_KEY"])
        self.model = config.get("model", "text-embedding-3-small")
        self.batch_size = config.get("batch_size", 100)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([e.embedding for e in response.data])
        return all_embeddings


class HuggingFaceEmbedding(BaseEmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        from sentence_transformers import SentenceTransformer
        model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = config.get("device", "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = config.get("batch_size", 32)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        return embeddings.tolist()


class CohereEmbedding(BaseEmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        import cohere
        self.client = cohere.Client(config.get("api_key") or os.environ["COHERE_API_KEY"])
        self.model = config.get("model", "embed-english-v3.0")
        self.batch_size = config.get("batch_size", 96)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embed(texts=batch, model=self.model, input_type="search_document")
            all_embeddings.extend(response.embeddings)
        return all_embeddings


class OllamaEmbedding(BaseEmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        import ollama
        self.client = ollama.Client(host=config.get("base_url", "http://localhost:11434"))
        self.model = config.get("model", "nomic-embed-text")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings


class AzureOpenAIEmbedding(BaseEmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=config.get("api_key") or os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=config.get("azure_endpoint") or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=config.get("api_version", "2024-02-01"),
        )
        self.model = config.get("azure_deployment", config.get("model", "text-embedding-3-small"))
        self.batch_size = config.get("batch_size", 100)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([e.embedding for e in response.data])
        return all_embeddings


EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbedding,
    "huggingface": HuggingFaceEmbedding,
    "cohere": CohereEmbedding,
    "ollama": OllamaEmbedding,
    "azure_openai": AzureOpenAIEmbedding,
}


def get_embedding_provider(config: Dict[str, Any]) -> BaseEmbeddingProvider:
    provider_name = config.get("provider", "openai")
    cls = EMBEDDING_PROVIDERS.get(provider_name)
    if not cls:
        raise ValueError(f"Unknown embedding provider: {provider_name}. Choose from {list(EMBEDDING_PROVIDERS.keys())}")
    logger.info(f"Using embedding provider: {provider_name} / {config.get('model')}")
    return cls(config)


# ─────────────────────────────────────────────
# Vector Store Backends
# ─────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_collection(self):
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {}


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        import chromadb
        host = config.get("host", "localhost")
        port = config.get("port", 8000)
        if host in ("localhost", "127.0.0.1") and port == 8000:
            self.client = chromadb.Client()
        else:
            self.client = chromadb.HttpClient(host=host, port=port)
        self.collection_name = config.get("collection", "rag_documents")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        if filters:
            kwargs["where"] = filters
        results = self.collection.query(**kwargs)
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # convert distance to similarity
            })
        return output

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

    def get_stats(self) -> Dict[str, Any]:
        return {"count": self.collection.count(), "name": self.collection_name}


class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        import weaviate
        self.client = weaviate.connect_to_custom(
            http_host=config.get("host", "localhost"),
            http_port=config.get("port", 8080),
            http_secure=config.get("secure", False),
            grpc_host=config.get("host", "localhost"),
            grpc_port=config.get("grpc_port", 50051),
        )
        self.collection_name = config.get("collection", "RagDocuments").capitalize()
        self._ensure_collection()

    def _ensure_collection(self):
        import weaviate.classes as wvc
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
        self.collection = self.client.collections.get(self.collection_name)

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        import weaviate.classes as wvc
        objects = []
        for chunk, embedding in zip(chunks, embeddings):
            obj = wvc.data.DataObject(
                properties={"content": chunk["content"], "source": chunk.get("source", ""), **chunk.get("metadata", {})},
                vector=embedding,
                uuid=chunk["chunk_id"][:36] if len(chunk["chunk_id"]) >= 36 else None,
            )
            objects.append(obj)
        self.collection.data.insert_many(objects)
        return len(objects)

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        import weaviate.classes.query as wq
        results = self.collection.query.near_vector(near_vector=query_embedding, limit=top_k, return_metadata=wq.MetadataQuery(distance=True))
        output = []
        for obj in results.objects:
            output.append({
                "content": obj.properties.get("content", ""),
                "metadata": {k: v for k, v in obj.properties.items() if k != "content"},
                "score": 1 - obj.metadata.distance,
            })
        return output

    def delete_collection(self):
        self.client.collections.delete(self.collection_name)


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.client = QdrantClient(
            host=config.get("host", "localhost"),
            port=config.get("port", 6333),
            api_key=config.get("api_key"),
        )
        self.collection_name = config.get("collection", "rag_documents")
        self.dimensions = config.get("dimensions", 1536)
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        from qdrant_client.models import PointStruct
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={"content": chunk["content"], "chunk_id": chunk["chunk_id"], "source": chunk.get("source", ""), **chunk.get("metadata", {})},
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        results = self.client.search(collection_name=self.collection_name, query_vector=query_embedding, limit=top_k)
        return [{"content": r.payload.get("content", ""), "metadata": r.payload, "score": r.score} for r in results]

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.get("api_key") or os.environ["PINECONE_API_KEY"])
        index_name = config.get("index_name", "rag-index")
        self.index = pc.Index(index_name)
        self.namespace = config.get("namespace", "default")

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {"content": chunk["content"], "source": chunk.get("source", ""), **chunk.get("metadata", {})},
            })
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i:i+100], namespace=self.namespace)
        return len(vectors)

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        kwargs = {"vector": query_embedding, "top_k": top_k, "namespace": self.namespace, "include_metadata": True}
        if filters:
            kwargs["filter"] = filters
        results = self.index.query(**kwargs)
        return [{"content": m.metadata.get("content", ""), "metadata": m.metadata, "score": m.score} for m in results.matches]

    def delete_collection(self):
        self.index.delete(delete_all=True, namespace=self.namespace)


VECTOR_STORE_MAP = {
    "chroma": ChromaVectorStore,
    "weaviate": WeaviateVectorStore,
    "qdrant": QdrantVectorStore,
    "pinecone": PineconeVectorStore,
}


def get_vector_store(config: Dict[str, Any]) -> BaseVectorStore:
    provider = config.get("provider", "chroma")
    cls = VECTOR_STORE_MAP.get(provider)
    if not cls:
        raise ValueError(f"Unknown vector store: {provider}. Choose from {list(VECTOR_STORE_MAP.keys())}")
    logger.info(f"Using vector store: {provider}")
    return cls(config)


# ─────────────────────────────────────────────
# Embedding Component Orchestrator
# ─────────────────────────────────────────────

class EmbeddingComponent:
    def __init__(self, embedding_config: Dict[str, Any], vector_store_config: Dict[str, Any]):
        self.embedder = get_embedding_provider(embedding_config)
        self.vector_store = get_vector_store(vector_store_config)

    def process(self, chunks_path: str) -> Dict[str, Any]:
        """Load chunks JSON, generate embeddings, store in vector DB."""
        with open(chunks_path) as f:
            chunks = json.load(f)

        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        logger.info(f"Storing embeddings in vector store...")
        count = self.vector_store.add_documents(chunks, embeddings)
        stats = self.vector_store.get_stats()

        result = {"embedded": count, "vector_store_stats": stats}
        logger.info(f"Embedding complete: {result}")
        return result

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embed_query(query)
        return self.vector_store.search(query_embedding, top_k=top_k, filters=filters)


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="RAG Embedding Component")
    parser.add_argument("--chunks-path", required=True, help="Path to chunks JSON from ingestion")
    parser.add_argument("--output-path", required=True, help="Output path for embedding stats JSON")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    component = EmbeddingComponent(cfg.get("embedding", {}), cfg.get("vector_store", {}))
    result = component.process(args.chunks_path)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
