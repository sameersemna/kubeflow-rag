FROM python:3.11-slim

LABEL maintainer="kubeflow-rag"
LABEL component="embedding"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/embedding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY components/embedding/ /app/components/embedding/
COPY components/__init__.py /app/components/__init__.py

# Cache HuggingFace models during build (optional)
# ARG HF_MODEL=sentence-transformers/all-MiniLM-L6-v2
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$HF_MODEL')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from components.embedding.embedding import get_embedding_provider; print('OK')"

ENTRYPOINT ["python", "-m", "components.embedding.embedding"]
