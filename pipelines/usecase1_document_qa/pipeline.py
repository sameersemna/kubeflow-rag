"""
Use Case 1: Document Q&A Pipeline
Kubeflow Pipeline for ingesting documents and answering questions with source citations.

Usage:
    python pipelines/usecase1_document_qa/pipeline.py \
        --config configs/config.yaml \
        --host https://your-kf-host
"""

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Artifact, Metrics
from typing import List
import yaml
import argparse
import io
from contextlib import redirect_stdout


# ─────────────────────────────────────────────
# Component Definitions (containerized steps)
# ─────────────────────────────────────────────

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pypdf", "python-docx", "beautifulsoup4", "requests", "pyyaml"],
)
def ingest_documents(
    sources: List[str],
    chunk_size: int,
    chunk_overlap: int,
    splitter_strategy: str,
    chunks: Output[Dataset],
) -> int:
    """Ingest documents from files/URLs and produce chunks."""
    import sys
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Inline ingestion logic (component is self-contained)
    from components.ingestion.ingestion import IngestionComponent

    config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "splitter": splitter_strategy}
    component = IngestionComponent(config)
    component.ingest_to_json(sources, chunks.path)

    with open(chunks.path) as f:
        data = json.load(f)
    logger.info(f"Ingestion complete: {len(data)} chunks")
    return len(data)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "openai>=1.0.0", "chromadb", "sentence-transformers", "pyyaml", "numpy"
    ],
)
def embed_and_index(
    chunks: Input[Dataset],
    embedding_provider: str,
    embedding_model: str,
    vector_store_provider: str,
    vector_store_host: str,
    vector_store_collection: str,
    embedding_stats: Output[Artifact],
) -> int:
    """Embed chunks and store in vector database."""
    import json
    import os
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    embedding_config = {
        "provider": embedding_provider,
        "model": embedding_model,
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
    vector_store_config = {
        "provider": vector_store_provider,
        "host": vector_store_host,
        "collection": vector_store_collection,
    }

    from components.embedding.embedding import EmbeddingComponent
    component = EmbeddingComponent(embedding_config, vector_store_config)
    result = component.process(chunks.path)

    Path(embedding_stats.path).parent.mkdir(parents=True, exist_ok=True)
    with open(embedding_stats.path, "w") as f:
        json.dump(result, f, indent=2)

    return result.get("embedded", 0)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "openai>=1.0.0", "chromadb", "sentence-transformers", "pyyaml", "numpy"
    ],
)
def retrieve_context(
    query: str,
    chunks: Input[Dataset],
    embedding_provider: str,
    embedding_model: str,
    vector_store_provider: str,
    vector_store_host: str,
    vector_store_collection: str,
    retrieval_strategy: str,
    top_k: int,
    retrieval_results: Output[Dataset],
) -> int:
    """Retrieve relevant document chunks for the query."""
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)

    embedding_config = {
        "provider": embedding_provider,
        "model": embedding_model,
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
    vector_store_config = {
        "provider": vector_store_provider,
        "host": vector_store_host,
        "collection": vector_store_collection,
    }
    retrieval_config = {"strategy": retrieval_strategy, "top_k": top_k}

    from components.embedding.embedding import EmbeddingComponent
    from components.retrieval.retrieval import RetrievalComponent

    with open(chunks.path) as f:
        corpus = json.load(f)

    emb_component = EmbeddingComponent(embedding_config, vector_store_config)
    ret_component = RetrievalComponent(retrieval_config, embedding_component=emb_component, corpus=corpus)
    ret_component.retrieve_and_save(query, retrieval_results.path, top_k=top_k)

    with open(retrieval_results.path) as f:
        data = json.load(f)
    return len(data.get("results", []))


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["openai>=1.0.0", "anthropic", "pyyaml"],
)
def generate_answer(
    query: str,
    retrieval_results: Input[Dataset],
    llm_provider: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    include_sources: bool,
    generation_output: Output[Dataset],
) -> str:
    """Generate an answer using the LLM and retrieved context."""
    import json
    import os
    import logging

    logging.basicConfig(level=logging.INFO)

    generation_config = {
        "provider": llm_provider,
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_sources": include_sources,
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }

    from components.generation.generation import GenerationComponent
    component = GenerationComponent(generation_config)
    component.process_from_files(query, retrieval_results.path, generation_output.path)

    with open(generation_output.path) as f:
        data = json.load(f)
    return data.get("answer", "")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pyyaml"],
)
def evaluate_rag(
    generation_output: Input[Dataset],
    retrieval_results: Input[Dataset],
    evaluation_framework: str,
    faithfulness_threshold: float,
    relevancy_threshold: float,
    evaluation_report: Output[Artifact],
    metrics: Output[Metrics],
) -> bool:
    """Evaluate the RAG pipeline quality."""
    import json
    import logging

    logging.basicConfig(level=logging.INFO)

    evaluation_config = {
        "framework": evaluation_framework,
        "thresholds": {
            "keyword_faithfulness": faithfulness_threshold,
            "answer_relevancy": relevancy_threshold,
        },
    }

    from components.evaluation.evaluation import EvaluationComponent
    component = EvaluationComponent(evaluation_config)
    component.process_from_files(
        generation_output.path,
        retrieval_results.path,
        evaluation_report.path,
    )

    with open(evaluation_report.path) as f:
        report = json.load(f)

    # Log metrics to KFP
    metrics.log_metric("overall_score", report["overall_score"])
    for result in report.get("results", []):
        metrics.log_metric(result["metric"], result["score"])

    return report.get("passed", False)


# ─────────────────────────────────────────────
# Pipeline Definition
# ─────────────────────────────────────────────

@dsl.pipeline(
    name="Document Q&A RAG Pipeline",
    description="Ingest documents, embed, retrieve, and generate answers with source citations.",
)
def document_qa_pipeline(
    # Input documents
    document_sources: List[str] = ["https://example.com/docs"],
    # Query
    query: str = "What is the main topic of the documents?",
    # Ingestion params
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_strategy: str = "recursive",
    # Embedding params
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    # Vector store params
    vector_store_provider: str = "chroma",
    vector_store_host: str = "chroma-service",
    vector_store_collection: str = "doc_qa_collection",
    # Retrieval params
    retrieval_strategy: str = "hybrid",
    top_k: int = 5,
    # Generation params
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    include_sources: bool = True,
    # Evaluation params
    evaluation_framework: str = "custom",
    faithfulness_threshold: float = 0.7,
    relevancy_threshold: float = 0.6,
):
    # Step 1: Ingest
    ingest_task = ingest_documents(
        sources=document_sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_strategy=splitter_strategy,
    )
    ingest_task.set_display_name("Ingest Documents")
    ingest_task.set_cpu_request("1").set_memory_request("2Gi")

    # Step 2: Embed & Index
    embed_task = embed_and_index(
        chunks=ingest_task.outputs["chunks"],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_provider=vector_store_provider,
        vector_store_host=vector_store_host,
        vector_store_collection=vector_store_collection,
    )
    embed_task.set_display_name("Embed & Index")
    embed_task.set_cpu_request("2").set_memory_request("4Gi")

    # Step 3: Retrieve
    retrieve_task = retrieve_context(
        query=query,
        chunks=ingest_task.outputs["chunks"],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_provider=vector_store_provider,
        vector_store_host=vector_store_host,
        vector_store_collection=vector_store_collection,
        retrieval_strategy=retrieval_strategy,
        top_k=top_k,
    )
    retrieve_task.set_display_name("Retrieve Context")
    retrieve_task.after(embed_task)  # Ensure indexing done before retrieval

    # Step 4: Generate
    generate_task = generate_answer(
        query=query,
        retrieval_results=retrieve_task.outputs["retrieval_results"],
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        include_sources=include_sources,
    )
    generate_task.set_display_name("Generate Answer")

    # Step 5: Evaluate
    eval_task = evaluate_rag(
        generation_output=generate_task.outputs["generation_output"],
        retrieval_results=retrieve_task.outputs["retrieval_results"],
        evaluation_framework=evaluation_framework,
        faithfulness_threshold=faithfulness_threshold,
        relevancy_threshold=relevancy_threshold,
    )
    eval_task.set_display_name("Evaluate Quality")


def compile_pipeline(output_file: str = "document_qa_pipeline.yaml"):
    """Compile the pipeline to YAML for Kubeflow."""
    kfp.compiler.Compiler().compile(
        pipeline_func=document_qa_pipeline,
        package_path=output_file,
    )
    print(f"Pipeline compiled to: {output_file}")


def run_pipeline(host: str, config_path: str, experiment_name: str = "document-qa"):
    """Submit the pipeline to Kubeflow."""
    import yaml as _yaml
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)

    client = kfp.Client(host=host)

    with redirect_stdout(io.StringIO()):
        run = client.create_run_from_pipeline_func(
            document_qa_pipeline,
            arguments={
                "embedding_provider": cfg.get("embedding", {}).get("provider", "openai"),
                "embedding_model": cfg.get("embedding", {}).get("model", "text-embedding-3-small"),
                "vector_store_provider": cfg.get("vector_store", {}).get("provider", "chroma"),
                "vector_store_host": cfg.get("vector_store", {}).get("host", "chroma-service"),
                "retrieval_strategy": cfg.get("retrieval", {}).get("strategy", "hybrid"),
                "top_k": cfg.get("retrieval", {}).get("top_k", 5),
                "llm_provider": cfg.get("llm", {}).get("provider", "openai"),
                "llm_model": cfg.get("llm", {}).get("model", "gpt-4o"),
            },
            experiment_name=experiment_name,
        )
    print(f"Pipeline run created: {run.run_id}")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile pipeline to YAML")
    parser.add_argument("--run", action="store_true", help="Submit pipeline run")
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="pipelines/usecase1_document_qa/pipeline.yaml")
    parser.add_argument("--experiment", default="document-qa")
    args = parser.parse_args()

    if args.compile:
        compile_pipeline(args.output)
    elif args.run:
        run_pipeline(args.host, args.config, args.experiment)
    else:
        compile_pipeline(args.output)
