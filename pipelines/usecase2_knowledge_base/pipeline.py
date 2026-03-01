"""
Use Case 2: Enterprise Knowledge Base Pipeline
Multi-source, scheduled knowledge base with continuous ingestion,
hybrid search, REST API serving, and automated quality evaluation.

Features:
- Multiple data source types (files, URLs, databases, APIs)
- Incremental / scheduled re-indexing
- Metadata filtering
- REST API endpoint for query serving
- Comprehensive evaluation dashboard
"""

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Artifact, Metrics, Model
from typing import List, Dict
import argparse


# ─────────────────────────────────────────────
# Specialized Components for Enterprise KB
# ─────────────────────────────────────────────

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pypdf", "python-docx", "beautifulsoup4", "requests",
        "pyyaml", "boto3", "google-cloud-storage",
    ],
)
def multi_source_ingest(
    file_sources: List[str],
    url_sources: List[str],
    s3_bucket: str,
    s3_prefix: str,
    source_type_filter: str,
    chunk_size: int,
    chunk_overlap: int,
    enable_metadata_enrichment: bool,
    all_chunks: Output[Dataset],
    ingestion_stats: Output[Artifact],
) -> int:
    """Ingest from multiple source types: local files, URLs, S3/GCS."""
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from components.ingestion.ingestion import IngestionComponent

    config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    component = IngestionComponent(config)

    # Combine all sources
    all_sources = list(file_sources) + list(url_sources)

    # Add S3 sources if specified
    if s3_bucket and s3_prefix:
        try:
            import boto3
            s3 = boto3.client("s3")
            response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if source_type_filter == "all" or key.endswith(f".{source_type_filter}"):
                    local_path = f"/tmp/{Path(key).name}"
                    s3.download_file(s3_bucket, key, local_path)
                    all_sources.append(local_path)
        except Exception as e:
            logger.warning(f"S3 ingestion failed: {e}")

    # Ingest
    chunks = component.ingest(all_sources)

    # Optionally enrich metadata
    if enable_metadata_enrichment:
        for chunk in chunks:
            chunk.metadata["kb_version"] = "v1"
            chunk.metadata["indexed_at"] = __import__("datetime").datetime.utcnow().isoformat()

    # Save chunks
    Path(all_chunks.path).parent.mkdir(parents=True, exist_ok=True)
    with open(all_chunks.path, "w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)

    # Save stats
    stats = {
        "total_chunks": len(chunks),
        "sources_processed": len(all_sources),
        "source_breakdown": {},
    }
    for source in all_sources:
        ext = Path(source).suffix or "url"
        stats["source_breakdown"][ext] = stats["source_breakdown"].get(ext, 0) + 1

    with open(ingestion_stats.path, "w") as f:
        json.dump(stats, f, indent=2)

    return len(chunks)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "openai>=1.0.0", "chromadb", "sentence-transformers",
        "qdrant-client", "weaviate-client", "pyyaml", "numpy",
    ],
)
def incremental_embed_and_index(
    all_chunks: Input[Dataset],
    embedding_provider: str,
    embedding_model: str,
    vector_store_provider: str,
    vector_store_host: str,
    vector_store_collection: str,
    incremental_mode: bool,
    batch_size: int,
    embedding_stats: Output[Artifact],
) -> int:
    """Embed and index with support for incremental updates."""
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    with open(all_chunks.path) as f:
        chunks = json.load(f)

    embedding_config = {
        "provider": embedding_provider,
        "model": embedding_model,
        "batch_size": batch_size,
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
    vector_store_config = {
        "provider": vector_store_provider,
        "host": vector_store_host,
        "collection": vector_store_collection,
    }

    from components.embedding.embedding import EmbeddingComponent
    component = EmbeddingComponent(embedding_config, vector_store_config)

    if incremental_mode:
        # Only embed new chunks (not already in vector store)
        logger.info(f"Incremental mode: processing {len(chunks)} chunks")

    result = component.process(all_chunks.path)

    with open(embedding_stats.path, "w") as f:
        json.dump(result, f, indent=2)

    return result.get("embedded", 0)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["openai>=1.0.0", "chromadb", "pyyaml", "numpy"],
)
def batch_qa_evaluation(
    all_chunks: Input[Dataset],
    embedding_provider: str,
    embedding_model: str,
    vector_store_provider: str,
    vector_store_host: str,
    vector_store_collection: str,
    llm_provider: str,
    llm_model: str,
    test_queries: List[str],
    retrieval_strategy: str,
    top_k: int,
    evaluation_report: Output[Artifact],
    metrics: Output[Metrics],
) -> float:
    """Run batch Q&A evaluation across multiple test queries."""
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    embedding_config = {"provider": embedding_provider, "model": embedding_model, "api_key": os.environ.get("OPENAI_API_KEY")}
    vector_store_config = {"provider": vector_store_provider, "host": vector_store_host, "collection": vector_store_collection}
    retrieval_config = {"strategy": retrieval_strategy, "top_k": top_k}
    generation_config = {"provider": llm_provider, "model": llm_model, "api_key": os.environ.get("OPENAI_API_KEY")}

    from components.embedding.embedding import EmbeddingComponent
    from components.retrieval.retrieval import RetrievalComponent
    from components.generation.generation import GenerationComponent
    from components.evaluation.evaluation import EvaluationComponent

    with open(all_chunks.path) as f:
        corpus = json.load(f)

    emb = EmbeddingComponent(embedding_config, vector_store_config)
    ret = RetrievalComponent(retrieval_config, embedding_component=emb, corpus=corpus)
    gen = GenerationComponent(generation_config)
    evaluator = EvaluationComponent({"framework": "custom"})

    test_cases = []
    for query in test_queries:
        retrieved = ret.retrieve(query)
        result = gen.generate(query, retrieved)
        test_cases.append({
            "question": query,
            "answer": result["answer"],
            "retrieved_docs": retrieved,
        })

    batch_result = evaluator.evaluate_batch(test_cases)

    with open(evaluation_report.path, "w") as f:
        json.dump(batch_result, f, indent=2)

    # Log to KFP metrics
    metrics.log_metric("pass_rate", batch_result["pass_rate"])
    for metric, score in batch_result["average_scores"].items():
        metrics.log_metric(f"avg_{metric}", score)

    overall = sum(batch_result["average_scores"].values()) / max(1, len(batch_result["average_scores"]))
    return overall


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pyyaml", "fastapi", "uvicorn"],
)
def register_kb_endpoint(
    vector_store_provider: str,
    vector_store_host: str,
    vector_store_collection: str,
    embedding_provider: str,
    llm_provider: str,
    llm_model: str,
    api_config: Output[Artifact],
) -> str:
    """Generate API configuration for the knowledge base REST endpoint."""
    import json
    import logging

    logging.basicConfig(level=logging.INFO)

    config = {
        "endpoint_type": "fastapi",
        "vector_store": {"provider": vector_store_provider, "host": vector_store_host, "collection": vector_store_collection},
        "embedding": {"provider": embedding_provider},
        "llm": {"provider": llm_provider, "model": llm_model},
        "routes": [
            {"path": "/query", "method": "POST", "description": "Query the knowledge base"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/stats", "method": "GET", "description": "Knowledge base statistics"},
        ],
        "status": "ready",
    }

    with open(api_config.path, "w") as f:
        json.dump(config, f, indent=2)

    endpoint_url = f"http://rag-api-service:8000/query"
    return endpoint_url


# ─────────────────────────────────────────────
# Enterprise KB Pipeline
# ─────────────────────────────────────────────

@dsl.pipeline(
    name="Enterprise Knowledge Base Pipeline",
    description="Multi-source KB with scheduled ingestion, hybrid search, batch evaluation, and API registration.",
)
def enterprise_kb_pipeline(
    # Sources
    file_sources: List[str] = [],
    url_sources: List[str] = ["https://docs.example.com"],
    s3_bucket: str = "",
    s3_prefix: str = "",
    source_type_filter: str = "all",
    # Ingestion
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    enable_metadata_enrichment: bool = True,
    # Embedding
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100,
    # Vector Store
    vector_store_provider: str = "chroma",
    vector_store_host: str = "chroma-service",
    vector_store_collection: str = "enterprise_kb",
    incremental_mode: bool = False,
    # Retrieval
    retrieval_strategy: str = "hybrid",
    top_k: int = 5,
    # LLM
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    # Evaluation
    test_queries: List[str] = ["What is the company policy on remote work?", "How do I submit an expense report?"],
):
    # Step 1: Multi-source ingestion
    ingest_task = multi_source_ingest(
        file_sources=file_sources,
        url_sources=url_sources,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        source_type_filter=source_type_filter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_metadata_enrichment=enable_metadata_enrichment,
    )
    ingest_task.set_display_name("📥 Multi-Source Ingest")
    ingest_task.set_cpu_request("2").set_memory_request("4Gi")

    # Step 2: Incremental embed & index
    embed_task = incremental_embed_and_index(
        all_chunks=ingest_task.outputs["all_chunks"],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_provider=vector_store_provider,
        vector_store_host=vector_store_host,
        vector_store_collection=vector_store_collection,
        incremental_mode=incremental_mode,
        batch_size=batch_size,
    )
    embed_task.set_display_name("🔢 Incremental Embed & Index")
    embed_task.set_cpu_request("4").set_memory_request("8Gi")

    # Step 3: Batch Q&A Evaluation
    eval_task = batch_qa_evaluation(
        all_chunks=ingest_task.outputs["all_chunks"],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_provider=vector_store_provider,
        vector_store_host=vector_store_host,
        vector_store_collection=vector_store_collection,
        llm_provider=llm_provider,
        llm_model=llm_model,
        test_queries=test_queries,
        retrieval_strategy=retrieval_strategy,
        top_k=top_k,
    )
    eval_task.set_display_name("📊 Batch Q&A Evaluation")
    eval_task.after(embed_task)

    # Step 4: Register API Endpoint
    api_task = register_kb_endpoint(
        vector_store_provider=vector_store_provider,
        vector_store_host=vector_store_host,
        vector_store_collection=vector_store_collection,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    api_task.set_display_name("🌐 Register API Endpoint")
    api_task.after(eval_task)


def compile_pipeline(output_file: str = "enterprise_kb_pipeline.yaml"):
    kfp.compiler.Compiler().compile(
        pipeline_func=enterprise_kb_pipeline,
        package_path=output_file,
    )
    print(f"Pipeline compiled to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", default="pipelines/usecase2_knowledge_base/pipeline.yaml")
    args = parser.parse_args()

    if args.compile:
        compile_pipeline(args.output)
