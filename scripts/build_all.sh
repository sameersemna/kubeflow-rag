#!/bin/bash
# ============================================================
# Build all RAG pipeline Docker images
# Usage: ./scripts/build_all.sh --registry my-registry.io --tag v1.0.0
# ============================================================

set -euo pipefail

REGISTRY="localhost"
TAG="latest"
PUSH=false
PARALLEL=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --push) PUSH=true; shift ;;
        --parallel) PARALLEL=true; shift ;;
        --help)
            echo "Usage: $0 [--registry REGISTRY] [--tag TAG] [--push] [--parallel]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

COMPONENTS=("ingestion" "embedding" "retrieval" "generation" "evaluation" "ui")

echo "🐳 Building Kubeflow RAG Docker images"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"
echo "Components: ${COMPONENTS[*]}"
echo ""

build_image() {
    local component=$1
    local image_name="$REGISTRY/rag-$component:$TAG"
    local dockerfile="docker/$component/Dockerfile"

    if [ ! -f "$dockerfile" ]; then
        echo "⚠️  Dockerfile not found: $dockerfile — skipping"
        return 0
    fi

    echo "🔨 Building $component → $image_name"
    docker build --file "$dockerfile" --tag "$image_name" --tag "$REGISTRY/rag-$component:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from "$image_name" . # 2>&1 | tail -5

    echo "✅ Built: $image_name"

    if [ "$PUSH" = true ]; then
        echo "📤 Pushing $image_name..."
        docker push "$image_name"
        docker push "$REGISTRY/rag-$component:latest"
        echo "✅ Pushed: $image_name"
    fi
}

export -f build_image
export REGISTRY TAG PUSH

if [ "$PARALLEL" = true ]; then
    echo "Running builds in parallel..."
    printf '%s\n' "${COMPONENTS[@]}" | xargs -P 4 -I {} bash -c 'build_image "$@"' _ {}
else
    for component in "${COMPONENTS[@]}"; do
        build_image "$component"
    done
fi

echo ""
echo "✨ All builds complete!"
echo ""
echo "Images:"
for component in "${COMPONENTS[@]}"; do
    echo ">  $REGISTRY/rag-$component:$TAG"
done
