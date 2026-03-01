#!/usr/bin/env bash

set -euo pipefail

KFP_VERSION="${KFP_VERSION:-2.2.0}"
KFP_NAMESPACE="${KFP_NAMESPACE:-kubeflow}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-600s}"
START_PORT_FORWARD="${START_PORT_FORWARD:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      KFP_VERSION="$2"
      shift 2
      ;;
    --namespace)
      KFP_NAMESPACE="$2"
      shift 2
      ;;
    --timeout)
      ROLLOUT_TIMEOUT="$2"
      shift 2
      ;;
    --port-forward)
      START_PORT_FORWARD="true"
      shift
      ;;
    --help)
      cat <<'EOF'
Usage: scripts/setup_kfp_local.sh [options]

Options:
  --version <ver>       KFP manifests version (default: 2.2.0)
  --namespace <name>    Namespace for KFP installation (default: kubeflow)
  --timeout <duration>  Rollout timeout, e.g. 600s (default: 600s)
  --port-forward        Start local port-forward to KFP API on localhost:8080
  --help                Show this help

Environment variables:
  KFP_VERSION, KFP_NAMESPACE, ROLLOUT_TIMEOUT, START_PORT_FORWARD
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "==> Installing Kubeflow Pipelines ${KFP_VERSION} (cluster-scoped resources)"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KFP_VERSION}"

echo "==> Installing Kubeflow Pipelines ${KFP_VERSION} (platform-agnostic resources)"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KFP_VERSION}"

echo "==> Applying local compatibility patches"
kubectl -n "${KFP_NAMESPACE}" set image deployment/minio minio=minio/minio:RELEASE.2019-08-14T20-37-41Z
kubectl -n "${KFP_NAMESPACE}" set image deployment/mysql mysql=mysql:5.7

echo "==> Recreating mysql PVC for mysql:5.7 compatibility"
kubectl -n "${KFP_NAMESPACE}" scale deployment/mysql --replicas=0
kubectl -n "${KFP_NAMESPACE}" delete pvc mysql-pv-claim --ignore-not-found
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KFP_VERSION}"
kubectl -n "${KFP_NAMESPACE}" set image deployment/mysql mysql=mysql:5.7
kubectl -n "${KFP_NAMESPACE}" scale deployment/mysql --replicas=1

echo "==> Waiting for core deployments"
kubectl -n "${KFP_NAMESPACE}" rollout status deployment/mysql --timeout="${ROLLOUT_TIMEOUT}"
kubectl -n "${KFP_NAMESPACE}" rollout status deployment/minio --timeout="${ROLLOUT_TIMEOUT}"
kubectl -n "${KFP_NAMESPACE}" rollout restart deployment/ml-pipeline
kubectl -n "${KFP_NAMESPACE}" rollout status deployment/ml-pipeline --timeout="${ROLLOUT_TIMEOUT}"

echo "==> Core services"
kubectl -n "${KFP_NAMESPACE}" get svc ml-pipeline mysql minio-service

if [[ "${START_PORT_FORWARD}" == "true" ]]; then
  echo "==> Starting port-forward: localhost:8080 -> svc/ml-pipeline:8888"
  echo "    Press Ctrl+C to stop"
  kubectl -n "${KFP_NAMESPACE}" port-forward svc/ml-pipeline 8080:8888
else
  cat <<EOF

Setup complete.
To access KFP API locally, run:
  kubectl -n ${KFP_NAMESPACE} port-forward svc/ml-pipeline 8080:8888

Then submit pipelines with:
  python scripts/run_pipeline.py --pipeline pipelines/usecase1_document_qa/pipeline.py --config configs/config.yaml --experiment "document-qa-experiment" --namespace ${KFP_NAMESPACE} --strict-preflight
EOF
fi
