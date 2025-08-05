#!/bin/bash

# Quantum MLOps Workbench - Production Deployment Script
# This script provides automated deployment to various environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
DEPLOY_TYPE="docker-compose"
NAMESPACE="quantum-mlops-prod"
IMAGE_TAG="latest"
DOMAIN="quantum-mlops.example.com"
AWS_REGION="us-east-1"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Environment (production|staging|development) [default: production]"
    echo "  -t, --type          Deployment type (docker-compose|kubernetes|cloud) [default: docker-compose]"
    echo "  -n, --namespace     Kubernetes namespace [default: quantum-mlops-prod]"
    echo "  -i, --image-tag     Docker image tag [default: latest]"
    echo "  -d, --domain        Domain name [default: quantum-mlops.example.com]"
    echo "  -r, --region        AWS region [default: us-east-1]"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --environment production --type kubernetes"
    echo "  $0 --type docker-compose --image-tag v1.0.0"
    echo "  $0 --environment staging --domain staging.quantum-mlops.com"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "git")
    
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        required_commands+=("kubectl" "helm")
    fi
    
    if [[ "$DEPLOY_TYPE" == "cloud" ]]; then
        required_commands+=("aws" "terraform")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build production image
    docker build -f Dockerfile.production --target production -t "quantum-mlops:${IMAGE_TAG}" .
    docker build -f Dockerfile.production --target worker -t "quantum-mlops-worker:${IMAGE_TAG}" .
    docker build -f Dockerfile.production --target scheduler -t "quantum-mlops-scheduler:${IMAGE_TAG}" .
    
    log_success "Docker images built successfully"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create environment file
    cat > .env <<EOF
ENVIRONMENT=${ENVIRONMENT}
IMAGE_TAG=${IMAGE_TAG}
DOMAIN=${DOMAIN}
AWS_DEFAULT_REGION=${AWS_REGION}
EOF
    
    # Deploy services
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.production.yml up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f "http://localhost:8000/health" &> /dev/null; then
        log_success "Quantum MLOps API is healthy"
    else
        log_warning "API health check failed, but deployment may still be starting"
    fi
    
    log_success "Docker Compose deployment completed"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image tags in deployment
    sed -i.bak "s|quantum-mlops:latest|quantum-mlops:${IMAGE_TAG}|g" k8s/production-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/production-deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/quantum-mlops-api -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/quantum-mlops-worker -n "$NAMESPACE"
    
    # Get service information
    kubectl get services -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
    
    # Restore original file
    mv k8s/production-deployment.yaml.bak k8s/production-deployment.yaml
}

deploy_cloud() {
    log_info "Deploying to cloud infrastructure..."
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Initialize Terraform
    cd terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="environment=${ENVIRONMENT}" -var="region=${AWS_REGION}" -var="domain=${DOMAIN}"
    
    # Apply deployment
    log_info "Applying Terraform configuration..."
    terraform apply -var="environment=${ENVIRONMENT}" -var="region=${AWS_REGION}" -var="domain=${DOMAIN}" -auto-approve
    
    cd ..
    
    log_success "Cloud deployment completed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        # Install Prometheus and Grafana using Helm
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo add grafana https://grafana.github.io/helm-charts
        helm repo update
        
        # Install Prometheus
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring --create-namespace \
            --values monitoring/prometheus-values.yaml
        
        log_success "Monitoring stack deployed to Kubernetes"
    else
        log_info "Monitoring is included in Docker Compose deployment"
    fi
}

run_tests() {
    log_info "Running deployment tests..."
    
    # Wait for services to be fully ready
    sleep 60
    
    # Test API endpoint
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        API_URL=$(kubectl get ingress quantum-mlops-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
        if [[ -z "$API_URL" ]]; then
            API_URL="localhost:8000"
        fi
    else
        API_URL="localhost:8000"
    fi
    
    # Health check
    if curl -f "http://${API_URL}/health" &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    # Test basic functionality
    if curl -f "http://${API_URL}/status" &> /dev/null; then
        log_success "API status check passed"
    else
        log_warning "API status check failed"
    fi
    
    log_success "Deployment tests completed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f .env
    log_success "Cleanup completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOY_TYPE="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ ! "$ENVIRONMENT" =~ ^(production|staging|development)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    usage
    exit 1
fi

if [[ ! "$DEPLOY_TYPE" =~ ^(docker-compose|kubernetes|cloud)$ ]]; then
    log_error "Invalid deployment type: $DEPLOY_TYPE"
    usage
    exit 1
fi

# Main deployment flow
main() {
    log_info "Starting Quantum MLOps Workbench deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOY_TYPE"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Domain: $DOMAIN"
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Execute deployment steps
    check_prerequisites
    build_images
    
    case $DEPLOY_TYPE in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        cloud)
            deploy_cloud
            ;;
    esac
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        setup_monitoring
    fi
    
    run_tests
    
    log_success "🚀 Quantum MLOps Workbench deployment completed successfully!"
    log_info "Access the API at: http://${DOMAIN}"
    log_info "Grafana dashboard: http://${DOMAIN}:3000 (admin/quantum123)"
    log_info "Prometheus: http://${DOMAIN}:9090"
}

# Run main function
main "$@"