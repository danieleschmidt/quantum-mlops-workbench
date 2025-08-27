#!/bin/bash

NAMESPACE="quantum-mlops"
SERVICE_NAME="quantum-meta-learning-service"

echo "🏥 Health Check for Quantum Meta-Learning"

# Check if pods are running
READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=quantum-meta-learning --field-selector=status.phase=Running --no-headers | wc -l)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l app=quantum-meta-learning --no-headers | wc -l)

echo "Pods Ready: $READY_PODS/$TOTAL_PODS"

# Check service endpoints
kubectl get endpoints $SERVICE_NAME -n $NAMESPACE

# Test API endpoint
if command -v curl &> /dev/null; then
    SERVICE_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    if curl -s http://$SERVICE_IP/health > /dev/null; then
        echo "✅ API Health Check: PASSED"
    else
        echo "❌ API Health Check: FAILED"
    fi
fi

echo "🔍 Recent logs:"
kubectl logs -l app=quantum-meta-learning -n $NAMESPACE --tail=10
