FROM python:3.11-slim

LABEL maintainer="kubeflow-rag"
LABEL component="ui-api"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY components/ /app/components/
COPY docker/ui/app.py /app/app.py
COPY configs/ /app/configs/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
