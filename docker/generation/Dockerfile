FROM python:3.11-slim

LABEL maintainer="kubeflow-rag"
LABEL component="generation"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/generation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY components/generation/ /app/components/generation/
COPY components/__init__.py /app/components/__init__.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from components.generation.generation import get_llm_provider; print('OK')"

ENTRYPOINT ["python", "-m", "components.generation.generation"]
