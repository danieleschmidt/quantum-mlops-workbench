#!/bin/bash
set -e

echo "🏗️ Building Quantum Meta-Learning Production Image"

# Build Docker image
docker build -t quantum-meta-learning:latest .
docker tag quantum-meta-learning:latest quantum-meta-learning:$(git rev-parse --short HEAD)

# Push to registry (if configured)
if [ ! -z "$DOCKER_REGISTRY" ]; then
    docker tag quantum-meta-learning:latest $DOCKER_REGISTRY/quantum-meta-learning:latest
    docker push $DOCKER_REGISTRY/quantum-meta-learning:latest
    docker push $DOCKER_REGISTRY/quantum-meta-learning:$(git rev-parse --short HEAD)
    echo "✅ Images pushed to registry"
fi

echo "✅ Build complete"
