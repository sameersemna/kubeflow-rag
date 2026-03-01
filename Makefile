# Makefile for Kubeflow RAG Template
# Usage: make help

REGISTRY ?= localhost
TAG ?= latest
CONFIG ?= configs/config.yaml
KF_HOST ?= http://localhost:8080
EXPERIMENT ?= rag-experiment
KFP_VERSION ?= 2.2.0
KFP_NAMESPACE ?= kubeflow
RUN_ID ?=

.PHONY: help install test lint format build push deploy-dev deploy-prod run-uc1 run-uc2 kfp-local kfp-local-forward submit-uc1-local submit-uc2-local watch-run clean

help:
	@echo "Kubeflow RAG Template - Available Commands"
	@echo "==========================================="
	@echo "  make install       - Install dev dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make lint          - Lint code"
	@echo "  make format        - Format code"
	@echo "  make build         - Build all Docker images"
	@echo "  make push          - Push images to registry"
	@echo "  make deploy-dev    - Deploy to K8s dev"
	@echo "  make deploy-prod   - Deploy to K8s production"
	@echo "  make run-uc1       - Run Use Case 1 locally"
	@echo "  make run-uc2       - Run Use Case 2 locally"
	@echo "  make kfp-local     - Install local KFP on kind (with compatibility patches)"
	@echo "  make kfp-local-forward - Install local KFP and start API port-forward"
	@echo "  make submit-uc1-local - Submit Use Case 1 to local KFP with strict preflight"
	@echo "  make submit-uc2-local - Submit Use Case 2 to local KFP with strict preflight"
	@echo "  make watch-run RUN_ID=<id> - Poll run status until completion"
	@echo "  make compile       - Compile KFP pipelines"
	@echo "  make up            - Start local services (docker-compose)"
	@echo "  make down          - Stop local services"
	@echo "  make clean         - Clean up build artifacts"

install:
	pip install -r requirements-dev.txt

test:
	pytest tests/unit/ -v --cov=components --cov-report=term-missing

lint:
	ruff check components/ pipelines/ examples/ scripts/
	black --check components/ pipelines/ examples/ scripts/

format:
	black components/ pipelines/ examples/ scripts/
	ruff check --fix components/ pipelines/ examples/ scripts/

build:
	./scripts/build_all.sh --registry $(REGISTRY) --tag $(TAG)

push:
	./scripts/build_all.sh --registry $(REGISTRY) --tag $(TAG) --push

deploy-dev:
	kubectl apply -k k8s/overlays/development/

deploy-prod:
	kubectl apply -k k8s/overlays/production/

run-uc1:
	python examples/usecase1_document_qa/run_local.py --config $(CONFIG)

run-uc2:
	python examples/usecase2_enterprise_kb/run_local.py --config $(CONFIG)

kfp-local:
	./scripts/setup_kfp_local.sh --version $(KFP_VERSION) --namespace $(KFP_NAMESPACE)

kfp-local-forward:
	./scripts/setup_kfp_local.sh --version $(KFP_VERSION) --namespace $(KFP_NAMESPACE) --port-forward

compile:
	python pipelines/usecase1_document_qa/pipeline.py --compile
	python pipelines/usecase2_knowledge_base/pipeline.py --compile
	@echo "Pipelines compiled!"

submit-uc1:
	python scripts/run_pipeline.py \
		--pipeline pipelines/usecase1_document_qa/pipeline.py \
		--config $(CONFIG) \
		--host $(KF_HOST) \
		--experiment $(EXPERIMENT)-uc1

submit-uc2:
	python scripts/run_pipeline.py \
		--pipeline pipelines/usecase2_knowledge_base/pipeline.py \
		--config $(CONFIG) \
		--host $(KF_HOST) \
		--experiment $(EXPERIMENT)-uc2

submit-uc1-local:
	python scripts/run_pipeline.py \
		--pipeline pipelines/usecase1_document_qa/pipeline.py \
		--config $(CONFIG) \
		--host $(KF_HOST) \
		--namespace $(KFP_NAMESPACE) \
		--experiment $(EXPERIMENT)-uc1 \
		--strict-preflight

submit-uc2-local:
	python scripts/run_pipeline.py \
		--pipeline pipelines/usecase2_knowledge_base/pipeline.py \
		--config $(CONFIG) \
		--host $(KF_HOST) \
		--namespace $(KFP_NAMESPACE) \
		--experiment $(EXPERIMENT)-uc2 \
		--strict-preflight

watch-run:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Usage: make watch-run RUN_ID=<run-id>"; \
		exit 1; \
	fi
	python scripts/watch_run.py --run-id $(RUN_ID) --host $(KF_HOST)

up:
	docker-compose up -d

up-full:
	docker-compose --profile full up -d

up-ollama:
	docker-compose --profile ollama up -d

down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.yaml" -path "*/compiled/*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
