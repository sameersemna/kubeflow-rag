FROM python:3.11-slim

LABEL maintainer="kubeflow-rag"
LABEL component="ingestion"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY docker/ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy component code
COPY components/ingestion/ /app/components/ingestion/
COPY components/__init__.py /app/components/__init__.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from components.ingestion.ingestion import IngestionComponent; print('OK')"

ENTRYPOINT ["python", "-m", "components.ingestion.ingestion"]
