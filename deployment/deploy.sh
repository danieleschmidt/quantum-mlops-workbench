#!/bin/bash
set -e

echo "🚀 Deploying Quantum Meta-Learning to Production"

# Create namespace if it doesn't exist
kubectl apply -f 01-namespace.yaml

# Apply all manifests
kubectl apply -f 02-configmap.yaml
kubectl apply -f 03-secrets.yaml
kubectl apply -f 07-pvc.yaml
kubectl apply -f 04-deployment.yaml
kubectl apply -f 05-service.yaml
kubectl apply -f 06-ingress.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/quantum-meta-learning -n quantum-mlops

echo "✅ Deployment complete"

# Display status
kubectl get pods -n quantum-mlops
kubectl get services -n quantum-mlops
