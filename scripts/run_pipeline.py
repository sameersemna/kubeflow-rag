#!/usr/bin/env python3
"""
Run a Kubeflow RAG pipeline.

Usage:
    python scripts/run_pipeline.py \
        --pipeline pipelines/usecase1_document_qa/pipeline.py \
        --config configs/config.yaml \
        --host https://your-kf-host \
        --experiment my-experiment
"""

import argparse
import importlib.util
import sys
import os
import yaml
from pathlib import Path


def load_pipeline_module(pipeline_path: str):
    spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run a Kubeflow RAG Pipeline")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline.py")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Kubeflow host URL")
    parser.add_argument("--experiment", default="rag-experiment", help="KFP experiment name")
    parser.add_argument("--compile-only", action="store_true", help="Only compile, don't run")
    parser.add_argument("--output", default=None, help="Output YAML path for compiled pipeline")
    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {args.config}")
    else:
        print(f"⚠️  Config not found at {args.config}, using defaults")

    # Resolve KFP host
    host = args.host or config.get("kubeflow", {}).get("host") or os.environ.get("KUBEFLOW_HOST")

    # Load pipeline module
    print(f"🔧 Loading pipeline: {args.pipeline}")
    module = load_pipeline_module(args.pipeline)

    # Compile
    output = args.output or str(Path(args.pipeline).with_suffix(".yaml"))
    if hasattr(module, "compile_pipeline"):
        module.compile_pipeline(output)
        print(f"📦 Pipeline compiled → {output}")
    else:
        print("⚠️  No compile_pipeline function found in module")

    if args.compile_only:
        print("Compile-only mode. Done.")
        return

    # Run
    if not host:
        print("❌ No Kubeflow host specified. Use --host or set KUBEFLOW_HOST env variable")
        print("   Running compile-only...")
        return

    if hasattr(module, "run_pipeline"):
        print(f"🚀 Submitting pipeline to {host}...")
        run = module.run_pipeline(host, args.config, args.experiment)
        print(f"✅ Pipeline run submitted! Run ID: {run.run_id}")
        print(f"   View at: {host}/#/runs/details/{run.run_id}")
    else:
        print("⚠️  No run_pipeline function found in module")


if __name__ == "__main__":
    main()
