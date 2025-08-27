#!/bin/bash
set -e

REVISION=${1:-1}

echo "🔄 Rolling back Quantum Meta-Learning deployment"

# Rollback deployment
kubectl rollout undo deployment/quantum-meta-learning --to-revision=$REVISION -n quantum-mlops

# Wait for rollback to complete
kubectl rollout status deployment/quantum-meta-learning -n quantum-mlops

echo "✅ Rollback to revision $REVISION complete"
