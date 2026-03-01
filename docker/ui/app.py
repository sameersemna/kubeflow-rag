"""
RAG API Server
FastAPI endpoint for serving the RAG pipeline as a REST API.
Supports streaming, filtering, and health checks.
"""

import os
import json
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kubeflow RAG API",
    description="REST API for the Kubeflow RAG Template pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question", min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False, description="Enable streaming response")
    collection: Optional[str] = Field(default=None, description="Override vector store collection")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]] = []
    retrieval_count: int = 0
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    vector_store_count: int
    collections: List[str]
    uptime_seconds: float


# ─────────────────────────────────────────────
# Startup / Dependencies
# ─────────────────────────────────────────────

_start_time = time.time()
_embedding_component = None
_retrieval_component = None
_generation_component = None
_config = {}


@app.on_event("startup")
async def startup_event():
    global _embedding_component, _retrieval_component, _generation_component, _config
    config_path = os.environ.get("CONFIG_PATH", "/app/configs/config.yaml")

    if Path(config_path).exists():
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using environment defaults")
        _config = {
            "embedding": {"provider": os.environ.get("EMBEDDING_PROVIDER", "openai"), "model": "text-embedding-3-small"},
            "vector_store": {"provider": os.environ.get("VECTOR_STORE_PROVIDER", "chroma"), "host": os.environ.get("VECTOR_STORE_HOST", "localhost")},
            "retrieval": {"strategy": "hybrid", "top_k": 5},
            "generation": {"provider": os.environ.get("LLM_PROVIDER", "openai"), "model": os.environ.get("LLM_MODEL", "gpt-4o")},
        }

    try:
        from components.embedding.embedding import EmbeddingComponent
        from components.retrieval.retrieval import RetrievalComponent
        from components.generation.generation import GenerationComponent

        _embedding_component = EmbeddingComponent(_config.get("embedding", {}), _config.get("vector_store", {}))
        _retrieval_component = RetrievalComponent(_config.get("retrieval", {}), embedding_component=_embedding_component)
        _generation_component = GenerationComponent(_config.get("generation", {}))
        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    components = {
        "embedding": "ok" if _embedding_component else "not_initialized",
        "retrieval": "ok" if _retrieval_component else "not_initialized",
        "generation": "ok" if _generation_component else "not_initialized",
    }
    status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=status, components=components)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not _retrieval_component or not _generation_component:
        raise HTTPException(status_code=503, detail="RAG components not initialized")

    start = time.time()
    try:
        # Retrieve
        retrieved = _retrieval_component.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )

        # Generate
        result = _generation_component.generate(request.query, retrieved)

        latency = (time.time() - start) * 1000
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result.get("sources", []),
            retrieval_count=len(retrieved),
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming endpoint for real-time response generation."""
    if not _retrieval_component or not _generation_component:
        raise HTTPException(status_code=503, detail="RAG components not initialized")

    retrieved = _retrieval_component.retrieve(query=request.query, top_k=request.top_k)

    def stream_generator():
        for chunk in _generation_component.generate_stream(request.query, retrieved):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    count = 0
    collections = []
    if _embedding_component:
        try:
            stats = _embedding_component.vector_store.get_stats()
            count = stats.get("count", 0)
            collections = [stats.get("name", "default")]
        except Exception:
            pass

    return StatsResponse(
        vector_store_count=count,
        collections=collections,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/")
async def root():
    return {
        "name": "Kubeflow RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
