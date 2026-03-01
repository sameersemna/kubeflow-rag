FROM python:3.11-slim

LABEL maintainer="kubeflow-rag"
LABEL component="evaluation"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY components/evaluation/ /app/components/evaluation/
COPY components/__init__.py /app/components/__init__.py

ENTRYPOINT ["python", "-m", "components.evaluation.evaluation"]
