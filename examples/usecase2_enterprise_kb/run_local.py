"""
Example: Enterprise Knowledge Base
Demonstrates a multi-source knowledge base with continuous ingestion,
hybrid search, and REST API serving.

Run locally:
    python examples/usecase2_enterprise_kb/run_local.py

Run on Kubeflow:
    python scripts/run_pipeline.py \
        --pipeline pipelines/usecase2_knowledge_base/pipeline.py \
        --config configs/config.yaml
"""

import os
import sys
import json
import logging
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_enterprise_knowledge_base(tmpdir: str):
    """Create a simulated enterprise knowledge base from multiple sources."""

    sources = {
        "hr_handbook.txt": """
HR Handbook - Global Operations Division

Onboarding Process:
New employees must complete I-9 verification within 3 days of hire.
All employees receive a laptop and equipment within 48 hours of start date.
The 90-day probationary review includes manager assessment and self-evaluation.
IT access is provisioned 24 hours before start date.

Performance Management:
Annual reviews occur in December for all permanent employees.
Quarterly check-ins are required for all direct reports.
Performance improvement plans (PIPs) require HR sign-off before initiation.
Promotions require 18 months in current role and two consecutive positive reviews.

Benefits Summary:
Health insurance: ACME covers 80% of premium for employee, 60% for dependents.
401k: Company matches up to 4% of salary, vesting after 2 years.
Stock options: Engineers receive 0.1-0.5% equity on a 4-year vesting schedule.
PTO: 15 days/year (years 0-2), 20 days/year (years 3-5), 25 days/year (5+).
        """,

        "it_runbook.md": """
# IT Operations Runbook

## VPN Setup
1. Download GlobalProtect from the company portal
2. Server address: vpn.acmecorp.com
3. Use your SSO credentials
4. Contact IT helpdesk if MFA token is lost

## Common Issues

### Laptop Won't Connect to Wi-Fi
- Forget network and reconnect
- Flush DNS: `sudo dscacheutil -flushcache`
- Restart networking: `sudo ifconfig en0 down && sudo ifconfig en0 up`

### Slack Not Loading
- Clear cache: rm -rf ~/Library/Application Support/Slack/Cache
- Try web app at acme.slack.com
- Check status.slack.com for outages

## Service Escalation Matrix
- P1 (Outage): Page on-call via PagerDuty immediately
- P2 (Major degradation): Slack #incidents, response within 1hr
- P3 (Minor issue): Submit ticket via helpdesk.acmecorp.com, SLA 24hr
- P4 (Request): Submit ticket, SLA 3 business days
        """,

        "finance_procedures.txt": """
Finance & Accounting Procedures

Accounts Payable:
All invoices must be submitted to ap@acmecorp.com with PO number.
Net-30 payment terms are standard unless negotiated otherwise.
Invoices over $10,000 require VP approval before payment.
Vendor payments are processed on the 15th and last business day of each month.

Budget Management:
Q1 budget planning begins October 1st for the following fiscal year.
Department heads must submit budget requests by November 15th.
Budget variances over 10% require CFO notification within 5 business days.
Capital expenditures over $50,000 require Board approval.

Travel & Entertainment:
Business travel requires manager approval 5 days in advance.
Economy class is required for flights under 5 hours.
Business class may be approved for flights over 8 hours with VP sign-off.
T&E reports must be submitted within 30 days of expense date.
Client entertainment is limited to $150/person without pre-approval.
        """,

        "engineering_standards.md": """
# Engineering Standards & Best Practices

## Code Review Process
All code must have at least 2 approvers before merging to main.
Security-sensitive changes require review from the security team.
PRs should be kept under 400 lines for reviewability.
All CI checks must pass before merge.

## Incident Response
1. Declare incident in #incidents Slack channel
2. Assign incident commander
3. Update status page every 15 minutes during P1/P2
4. Write post-mortem within 3 business days
5. Share learnings in all-hands engineering meeting

## Architecture Decision Records (ADRs)
All significant architectural decisions must be documented as ADRs.
ADRs are stored in the `docs/adr/` directory of each repository.
Template: docs/adr/0000-template.md

## API Design Standards
- Use RESTful conventions for new APIs
- Version all APIs (/v1/, /v2/, etc.)
- Return appropriate HTTP status codes
- Document all APIs using OpenAPI 3.0
- Rate limiting: implement for all public endpoints
        """,
    }

    paths = []
    for filename, content in sources.items():
        path = os.path.join(tmpdir, filename)
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)

    return paths


def simulate_api_queries(retrieval, gen_config, queries):
    """Simulate REST API queries against the knowledge base."""
    logger.info("\n🌐 Simulating REST API Queries...")
    logger.info("=" * 60)

    results = []
    for query in queries:
        start = time.time()
        retrieved = retrieval.retrieve(query, top_k=3)
        latency_ms = (time.time() - start) * 1000

        result = {
            "query": query,
            "retrieved_count": len(retrieved),
            "latency_ms": round(latency_ms, 2),
            "top_result": retrieved[0]["content"][:200] if retrieved else "No results",
            "sources": [r.get("metadata", {}).get("source", "unknown") for r in retrieved[:2]],
        }
        results.append(result)

        logger.info(f"\n❓ {query}")
        logger.info(f"   ⏱️  Retrieval: {latency_ms:.1f}ms")
        logger.info(f"   📄 Top result: {result['top_result'][:120]}...")
        logger.info(f"   📚 Sources: {result['sources']}")

    return results


def run_local_pipeline(config_path: str = None):
    """Run the Enterprise KB pipeline locally."""
    import yaml

    config = {
        "ingestion": {"chunk_size": 600, "chunk_overlap": 100},
        "embedding": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "vector_store": {
            "provider": "chroma",
            "host": "localhost",
            "collection": "enterprise_kb_demo",
        },
        "retrieval": {"strategy": "hybrid", "top_k": 3},
        "generation": {"provider": "openai", "model": "gpt-4o-mini"},
        "evaluation": {"framework": "custom"},
    }

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config.update(yaml.safe_load(f))

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("=" * 60)
        logger.info("Use Case 2: Enterprise Knowledge Base")
        logger.info("=" * 60)

        # Build knowledge base
        logger.info("\n📥 Phase 1: Multi-Source Ingestion")
        doc_paths = create_enterprise_knowledge_base(tmpdir)
        logger.info(f"Sources: {[Path(p).name for p in doc_paths]}")

        from components.ingestion.ingestion import IngestionComponent
        ingestion = IngestionComponent(config["ingestion"])
        chunks_path = os.path.join(tmpdir, "chunks.json")
        ingestion.ingest_to_json(doc_paths, chunks_path)

        with open(chunks_path) as f:
            chunks = json.load(f)
        logger.info(f"✅ Created {len(chunks)} chunks from {len(doc_paths)} sources")

        # Embed & Index
        logger.info("\n🔢 Phase 2: Embed & Index")
        from components.embedding.embedding import EmbeddingComponent
        embedding = EmbeddingComponent(config["embedding"], config["vector_store"])
        result = embedding.process(chunks_path)
        logger.info(f"✅ Indexed {result.get('embedded', 0)} vectors")

        # Set up retrieval
        from components.retrieval.retrieval import RetrievalComponent
        retrieval = RetrievalComponent(config["retrieval"], embedding_component=embedding, corpus=chunks)

        # Simulate API queries
        api_queries = [
            "What is the remote work policy?",
            "How do I set up VPN?",
            "What are the code review requirements?",
            "How do I submit an expense report?",
            "What are the promotion requirements?",
            "What is the incident response process?",
            "What are the travel expense limits?",
            "How does the 401k matching work?",
        ]

        api_results = simulate_api_queries(retrieval, config["generation"], api_queries)

        # Evaluate
        logger.info("\n📊 Phase 3: Knowledge Base Quality Evaluation")
        from components.evaluation.evaluation import EvaluationComponent
        evaluator = EvaluationComponent(config["evaluation"])

        test_cases = []
        for r in api_results:
            retrieved = retrieval.retrieve(r["query"], top_k=3)
            test_cases.append({
                "question": r["query"],
                "answer": r["top_result"],  # Use top retrieved as proxy
                "retrieved_docs": retrieved,
            })

        batch_result = evaluator.evaluate_batch(test_cases)

        logger.info("\n" + "=" * 60)
        logger.info("📈 KNOWLEDGE BASE QUALITY METRICS")
        logger.info("=" * 60)
        logger.info(f"Queries evaluated: {batch_result['total_cases']}")
        logger.info(f"Pass rate: {batch_result['pass_rate']:.1%}")
        logger.info(f"Avg latency: {sum(r['latency_ms'] for r in api_results) / len(api_results):.1f}ms")
        for metric, score in batch_result["average_scores"].items():
            logger.info(f"  {metric}: {score:.3f}")
        logger.info("=" * 60)

        # Show what the API config would look like
        api_config = {
            "endpoint": "http://rag-api-service:8000",
            "routes": ["/query", "/query/stream", "/health", "/stats"],
            "vector_store": config["vector_store"]["provider"],
            "embedding": config["embedding"]["model"],
            "total_documents": len(chunks),
            "status": "ready",
        }

        logger.info("\n🌐 API Configuration (ready to deploy):")
        logger.info(json.dumps(api_config, indent=2))
        logger.info("\n✅ Use Case 2 Demo Complete!")

        return {"evaluation": batch_result, "api_config": api_config}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_local_pipeline(args.config)
