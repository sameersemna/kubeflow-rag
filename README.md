# 🚀 Kubeflow RAG Template

A production-ready, reusable, and extensible **Retrieval-Augmented Generation (RAG)** framework built on **Kubeflow Pipelines**, designed for enterprise-scale AI/ML workloads.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Configuration](#configuration)
- [Customization](#customization)
- [Docker Images](#docker-images)
- [Kubeflow Pipelines](#kubeflow-pipelines)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## Overview

This template provides a **modular, scalable RAG pipeline** that runs on Kubeflow. It is designed to be:

- **Generic** – works with any document type, LLM, or vector store
- **Reusable** – fork and customize for your use case
- **Scalable** – add/swap components without rewriting the pipeline
- **Production-ready** – includes monitoring, logging, CI/CD, and testing

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubeflow Pipelines                           │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │Ingestion │→ │Embedding │→ │Retrieval │→ │   Generation     │   │
│  │Component │  │Component │  │Component │  │   Component      │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│        │             │             │                │               │
│        └─────────────┴─────────────┴────────────────┘               │
│                              │                                      │
│                    ┌──────────────────┐                             │
│                    │   Evaluation     │                             │
│                    │   Component      │                             │
│                    └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Core Components

| Component | Description | Swappable With |
|-----------|-------------|----------------|
| **Ingestion** | Load & chunk documents | LangChain, LlamaIndex, custom loaders |
| **Embedding** | Generate vector embeddings | OpenAI, HuggingFace, Cohere, Ollama |
| **Vector Store** | Store & index vectors | Chroma, Weaviate, Qdrant, Pinecone, Milvus |
| **Retrieval** | Semantic search & ranking | BM25, MMR, Hybrid Search |
| **Generation** | LLM response generation | GPT-4, Claude, Llama, Mistral, Gemini |
| **Evaluation** | RAG quality metrics | RAGAS, TruLens, custom metrics |

### Infrastructure

```
kubeflow-rag/
├── components/          # Reusable pipeline components (Python)
├── pipelines/           # Pipeline definitions (KFP DSL)
├── docker/              # Dockerfiles per component
├── k8s/                 # Kubernetes manifests (Kustomize)
├── kubeflow/            # KFP pipeline yamls & experiments
├── configs/             # Configuration files (YAML)
├── scripts/             # Utility scripts
├── examples/            # Two complete use case examples
├── tests/               # Unit & integration tests
└── docs/                # Extended documentation
```

---

## Features

- ✅ **Modular Pipeline Components** – each step is independently containerized
- ✅ **Multi-LLM Support** – OpenAI, Anthropic, HuggingFace, Ollama (local)
- ✅ **Multi-VectorStore Support** – Chroma, Weaviate, Qdrant, Pinecone, Milvus
- ✅ **Flexible Document Ingestion** – PDF, DOCX, HTML, CSV, JSON, Markdown, URLs
- ✅ **Hybrid Search** – dense + sparse (BM25) retrieval
- ✅ **RAG Evaluation** – automated quality metrics with RAGAS
- ✅ **Kubeflow Integration** – full KFP v2 DSL pipelines
- ✅ **Kubernetes-native** – Kustomize overlays for dev/prod
- ✅ **CI/CD Ready** – GitHub Actions workflows
- ✅ **Observability** – Prometheus metrics, structured logging
- ✅ **2 Complete Use Case Examples**

---

## Prerequisites

- Kubernetes cluster (v1.25+)
- Kubeflow Pipelines (v2.0+)
- `kubectl` configured
- `kfp` SDK installed (`pip install kfp>=2.0`)
- Docker
- Python 3.10+

---

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/sameersemna/kubeflow-rag.git
cd kubeflow-rag

# Copy and edit configuration
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your settings
```

### 2. Set Environment Variables

```bash
export KUBEFLOW_HOST="https://kubeflow.local"
export OPENAI_API_KEY="api-key"         # or other LLM provider
export VECTOR_STORE_HOST="localhost"
```

### 3. Build & Push Docker Images

```bash
# Build all images
./scripts/build_all.sh --registry docker.io --tag v1.0.0

# Push all images
./scripts/push_all.sh --registry docker.io --tag v1.0.0
```

### 4. Deploy Infrastructure

```bash
# Development
kubectl apply -k k8s/overlays/development/

# Production
kubectl apply -k k8s/overlays/production/
```

### 5. Run a Pipeline

```bash
# Run Use Case 1: Document Q&A
python scripts/run_pipeline.py \
  --pipeline pipelines/usecase1_document_qa/pipeline.py \
  --config configs/config.yaml \
  --experiment "document-qa-experiment"

# Run Use Case 2: Enterprise Knowledge Base
python scripts/run_pipeline.py \
  --pipeline pipelines/usecase2_knowledge_base/pipeline.py \
  --config configs/config.yaml \
  --experiment "enterprise-kb-experiment"
```

---

## Use Cases

### Use Case 1: Document Q&A System

A RAG pipeline that ingests documents (PDF/DOCX/HTML), creates a searchable knowledge base, and answers user questions with source citations.

**Pipeline:** `pipelines/usecase1_document_qa/`
**Example:** `examples/usecase1_document_qa/`

```
Documents → Ingest → Embed → Index → User Query → Retrieve → Generate Answer
```

### Use Case 2: Enterprise Knowledge Base

A multi-source knowledge base with continuous ingestion, scheduled re-indexing, and a REST API endpoint for integration.

**Pipeline:** `pipelines/usecase2_knowledge_base/`
**Example:** `examples/usecase2_enterprise_kb/`

```
Multiple Sources → Scheduled Ingestion → Embed → Hybrid Search → REST API → Answer
```

---

## Configuration

All configuration is centralized in `configs/config.yaml`:

```yaml
# configs/config.yaml
llm:
  provider: openai          # openai | anthropic | huggingface | ollama
  model: gpt-4o
  temperature: 0.1
  max_tokens: 2048

embedding:
  provider: openai          # openai | huggingface | cohere
  model: text-embedding-3-small
  dimensions: 1536

vector_store:
  provider: chroma          # chroma | weaviate | qdrant | pinecone | milvus
  host: localhost
  port: 8000
  collection: rag_documents

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: [pdf, docx, html, txt, md, csv, json]

retrieval:
  strategy: hybrid          # semantic | bm25 | hybrid | mmr
  top_k: 5
  score_threshold: 0.7

evaluation:
  enabled: true
  framework: ragas          # ragas | trulens | custom
  metrics: [faithfulness, answer_relevancy, context_precision]
```

---

## Customization

### Adding a New LLM Provider

1. Create `components/generation/providers/my_provider.py`
2. Implement the `BaseLLMProvider` interface
3. Register in `components/generation/factory.py`
4. Update `configs/config.yaml` with provider name

### Adding a New Vector Store

1. Create `components/embedding/stores/my_store.py`
2. Implement the `BaseVectorStore` interface
3. Register in `components/embedding/store_factory.py`
4. Add Docker service in `docker/` if needed

### Adding a New Pipeline Component

1. Create component in `components/my_component/`
2. Build Docker image in `docker/my_component/`
3. Add component spec in `kubeflow/components/`
4. Integrate into pipeline in `pipelines/`

---

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push and open a Pull Request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

---

## License

MIT License – see [LICENSE](LICENSE) for details.
