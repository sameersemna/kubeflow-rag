# Contributing to Kubeflow RAG Template

Thank you for your interest in contributing! This guide explains how to add new components, providers, and use cases.

---

## Adding a New LLM Provider

1. Open `components/generation/generation.py`
2. Create a new class inheriting from `BaseLLMProvider`:

```python
class MyLLMProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        # Initialize your client
        pass

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        # Call your LLM API
        return "response text"
```

3. Register it in `LLM_PROVIDERS` dict:
```python
LLM_PROVIDERS = {
    ...
    "my_provider": MyLLMProvider,
}
```

4. Update `configs/config.example.yaml` with usage example.

---

## Adding a New Vector Store

1. Open `components/embedding/embedding.py`
2. Create a class inheriting from `BaseVectorStore`
3. Register it in `VECTOR_STORE_MAP`
4. Add Docker service to `docker-compose.yml` if needed
5. Add K8s manifests in `k8s/base/` if needed

---

## Adding a New Pipeline Use Case

1. Create directory: `pipelines/usecaseN_my_pipeline/`
2. Create `pipeline.py` with:
   - `@dsl.pipeline` decorated function
   - `compile_pipeline()` function
   - `run_pipeline()` function
3. Create example: `examples/usecaseN_my_pipeline/run_local.py`
4. Add to README.md

---

## Adding a New Docker Image/Component

1. Create `docker/my_component/Dockerfile`
2. Create `docker/my_component/requirements.txt`
3. Create component code in `components/my_component/`
4. Add to `scripts/build_all.sh` COMPONENTS array
5. Add service to `docker-compose.yml`
6. Add K8s deployment if needed

---

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Write tests in `tests/unit/`
4. Run: `make lint && make test`
5. Submit a PR with description of changes

---

## Code Style

- Python: Black formatting (`make format`)
- Linting: Ruff (`make lint`)
- Type hints encouraged but not required
- Docstrings for all public methods
