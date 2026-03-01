"""
Example: Document Q&A System
Demonstrates a complete RAG pipeline for answering questions about documents.

Run locally (without Kubeflow):
    python examples/usecase1_document_qa/run_local.py

Run on Kubeflow:
    python scripts/run_pipeline.py \
        --pipeline pipelines/usecase1_document_qa/pipeline.py \
        --config configs/config.yaml
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_sample_documents(tmpdir: str):
    """Create sample documents for the demo."""
    docs = {
        "company_policy.txt": """
ACME Corporation Employee Policy Manual

Remote Work Policy:
Employees may work remotely up to 3 days per week with manager approval.
All remote work must be done in a secure environment with VPN connected.
Core hours for remote workers are 10am-3pm in their local timezone.

Expense Reporting:
All expenses over $50 must have a receipt.
Submit expenses via the company portal within 30 days of the expense.
Travel expenses require pre-approval for amounts exceeding $500.
Meals are reimbursable up to $75/day during business travel.

Vacation Policy:
Full-time employees earn 15 days PTO per year.
PTO must be approved 2 weeks in advance for trips longer than 3 days.
Unused PTO up to 5 days may be carried over to the next year.
        """,

        "product_faq.txt": """
ACME Widget Pro - Frequently Asked Questions

Q: What is the warranty period for ACME Widget Pro?
A: ACME Widget Pro comes with a 2-year limited warranty covering manufacturing defects.

Q: How do I reset the device?
A: Hold the reset button for 10 seconds until the LED flashes red, then release.

Q: Is the Widget Pro waterproof?
A: The Widget Pro has an IP67 rating and can withstand immersion in 1m of water for 30 minutes.

Q: What are the power requirements?
A: The Widget Pro operates on 110-240V AC power with the included adapter.

Q: How often should I update the firmware?
A: We recommend checking for firmware updates monthly via the ACME mobile app.
        """,

        "technical_docs.md": """
# ACME API Documentation v2.0

## Authentication
All API requests require an API key passed in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### GET /api/v2/widgets
Returns a list of all widgets.

**Parameters:**
- `limit` (int): Max results per page (default: 20, max: 100)
- `status` (string): Filter by status (active, inactive, all)

**Response:**
```json
{
  "widgets": [...],
  "total": 150,
  "page": 1
}
```

### POST /api/v2/widgets
Create a new widget.

**Rate Limits:**
- Free tier: 100 requests/hour
- Pro tier: 1000 requests/hour
- Enterprise: Unlimited

## Error Codes
- 400: Bad Request
- 401: Unauthorized
- 429: Rate limit exceeded
- 500: Server error
        """,
    }

    paths = []
    for filename, content in docs.items():
        path = os.path.join(tmpdir, filename)
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)
        logger.info(f"Created sample doc: {path}")

    return paths


def run_local_pipeline(config_path: str = None):
    """Run the Document Q&A pipeline locally without Kubeflow."""
    import yaml

    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Use defaults for demo
        config = {
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50},
            "embedding": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "vector_store": {
                "provider": "chroma",
                "host": "localhost",
                "collection": "demo_doc_qa",
            },
            "retrieval": {"strategy": "hybrid", "top_k": 3},
            "generation": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            "evaluation": {"framework": "custom"},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("=" * 60)
        logger.info("Use Case 1: Document Q&A System")
        logger.info("=" * 60)

        # Step 1: Create & Ingest Documents
        logger.info("\n📥 Step 1: Ingesting documents...")
        doc_paths = create_sample_documents(tmpdir)

        from components.ingestion.ingestion import IngestionComponent
        ingestion = IngestionComponent(config.get("ingestion", {}))
        chunks_path = os.path.join(tmpdir, "chunks.json")
        ingestion.ingest_to_json(doc_paths, chunks_path)

        with open(chunks_path) as f:
            chunks = json.load(f)
        logger.info(f"✅ Ingested {len(chunks)} chunks from {len(doc_paths)} documents")

        # Step 2: Embed & Index
        logger.info("\n🔢 Step 2: Embedding and indexing...")
        from components.embedding.embedding import EmbeddingComponent
        embedding = EmbeddingComponent(config.get("embedding", {}), config.get("vector_store", {}))
        result = embedding.process(chunks_path)
        logger.info(f"✅ Indexed {result.get('embedded', 0)} vectors")

        # Step 3-5: Q&A Loop
        test_queries = [
            "What is the remote work policy?",
            "How do I submit an expense report?",
            "What is the warranty on the Widget Pro?",
            "How do I reset the device?",
            "What are the API rate limits?",
        ]

        from components.retrieval.retrieval import RetrievalComponent
        from components.generation.generation import GenerationComponent
        from components.evaluation.evaluation import EvaluationComponent

        retrieval = RetrievalComponent(
            config.get("retrieval", {}),
            embedding_component=embedding,
            corpus=chunks,
        )

        eval_cases = []

        for query in test_queries:
            logger.info(f"\n❓ Query: {query}")

            # Retrieve
            retrieved = retrieval.retrieve(query)
            logger.info(f"🔍 Retrieved {len(retrieved)} relevant chunks")

            # Generate (skip if no API key set)
            answer = "[Generation skipped - set OPENAI_API_KEY to enable]"
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    gen = GenerationComponent(config.get("generation", {}))
                    result = gen.generate(query, retrieved)
                    answer = result["answer"]
                    logger.info(f"✨ Answer: {answer[:200]}...")
                    if result.get("sources"):
                        logger.info(f"📚 Sources: {[s['source'] for s in result['sources']]}")
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
            else:
                # Show what was retrieved
                if retrieved:
                    logger.info(f"📄 Top context: {retrieved[0]['content'][:200]}...")

            eval_cases.append({
                "question": query,
                "answer": answer,
                "retrieved_docs": retrieved,
            })

        # Step 5: Evaluate
        logger.info("\n📊 Step 5: Evaluating pipeline quality...")
        evaluator = EvaluationComponent(config.get("evaluation", {}))
        batch_result = evaluator.evaluate_batch(eval_cases)

        logger.info("\n" + "=" * 60)
        logger.info("📈 EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total queries: {batch_result['total_cases']}")
        logger.info(f"Pass rate: {batch_result['pass_rate']:.1%}")
        for metric, score in batch_result["average_scores"].items():
            logger.info(f"  {metric}: {score:.3f}")
        logger.info("=" * 60)
        logger.info("\n✅ Use Case 1 Demo Complete!")

        return batch_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_local_pipeline(args.config)
