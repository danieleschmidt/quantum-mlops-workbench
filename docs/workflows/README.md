# GitHub Actions Workflows

This directory contains production-ready GitHub Actions workflows for the Quantum MLOps platform. 

## 🚨 **IMPORTANT: Manual Setup Required**

Due to GitHub security restrictions, these workflow files need to be manually copied to `.github/workflows/` directory by a repository maintainer with appropriate permissions.

## Available Workflows

### 1. Quantum ML CI (`quantum-ml-ci.yml`)
- **Location**: `.github/workflows-templates/quantum-ml-ci.yml`
- **Purpose**: Comprehensive CI pipeline for quantum ML workloads
- **Features**:
  - Multi-framework testing (PennyLane, Qiskit, Cirq, AWS Braket)
  - Quantum hardware simulation and testing
  - Cross-platform compatibility (Linux, macOS, Windows)
  - Parallel test execution with pytest-xdist
  - Code quality checks with ruff, black, mypy
  - Coverage reporting and badge generation

### 2. Security Scanning (`security-scanning.yml`)
- **Location**: `.github/workflows-templates/security-scanning.yml`  
- **Purpose**: Comprehensive security analysis and SBOM generation
- **Features**:
  - Multi-scanner vulnerability detection (Safety, Bandit, Semgrep, OSV)
  - Container security with Trivy and Grype
  - SBOM generation in SPDX and CycloneDX formats
  - License compliance verification
  - Secret scanning and code analysis
  - Security advisory integration

### 3. Performance Benchmarking (`performance-benchmarking.yml`)
- **Location**: `.github/workflows-templates/performance-benchmarking.yml`
- **Purpose**: Quantum circuit performance analysis and benchmarking
- **Features**:
  - Circuit execution time measurements
  - Memory usage profiling
  - Quantum volume benchmarking
  - Performance regression detection
  - Hardware backend comparison
  - Benchmark result storage and trending

### 4. Release Automation (`release-automation.yml`)
- **Location**: `.github/workflows-templates/release-automation.yml`
- **Purpose**: Automated release management and deployment
- **Features**:
  - Semantic versioning with conventional commits
  - Automated changelog generation
  - Container image building and publishing
  - Package publishing to PyPI
  - Release notes generation
  - Deployment to staging and production

## Setup Instructions

### Step 1: Copy Workflow Files
```bash
# Copy all workflow templates to active workflows directory
cp .github/workflows-templates/*.yml .github/workflows/
```

### Step 2: Configure Secrets
Set up the following repository secrets in GitHub Settings > Secrets and variables > Actions:

#### Quantum Hardware Access
- `IBM_QUANTUM_TOKEN`: IBM Quantum Network API token
- `AWS_ACCESS_KEY_ID`: AWS access key for Braket
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for Braket
- `GOOGLE_CLOUD_CREDENTIALS`: Google Cloud service account JSON

#### Container Registry
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token
- `GHCR_TOKEN`: GitHub Container Registry token

#### PyPI Publishing
- `PYPI_API_TOKEN`: PyPI API token for package publishing

#### Security & Monitoring
- `SONAR_TOKEN`: SonarCloud token (optional)
- `CODECOV_TOKEN`: Codecov token for coverage reporting

### Step 3: Environment Configuration
Create environment protection rules for production deployments:
1. Go to Settings > Environments
2. Create `production` environment
3. Add required reviewers
4. Configure deployment branches (e.g., `main` only)

### Step 4: Configure Branch Protection
Enable branch protection on `main`:
1. Require pull request reviews
2. Require status checks to pass:
   - `test-quantum-backends`
   - `security-scan`
   - `build-and-test`
3. Require branches to be up to date
4. Include administrators

## Workflow Triggers

### Automatic Triggers
- **Push to main**: Full CI/CD pipeline
- **Pull requests**: Testing and security scans
- **Daily**: Security scans and dependency updates
- **Release tags**: Automated release and deployment

### Manual Triggers
- Performance benchmarking can be triggered manually
- Production deployment requires manual approval
- Emergency hotfix deployment workflow

## Monitoring and Alerts

All workflows integrate with:
- **Slack/Teams**: Build status notifications
- **Email**: Failure alerts for critical workflows  
- **GitHub**: Status checks and PR comments
- **Prometheus**: Workflow metrics collection

## Troubleshooting

### Common Issues

1. **Quantum Hardware Access Failures**
   - Verify API tokens are correct and not expired
   - Check quantum provider service status
   - Ensure sufficient credits/quota

2. **Container Build Failures**
   - Check Dockerfile syntax
   - Verify base image availability
   - Review layer caching issues

3. **Test Failures**
   - Review quantum simulator configuration
   - Check test dependencies
   - Verify quantum circuit compatibility

### Support

For workflow issues:
1. Check workflow run logs in GitHub Actions tab
2. Review this documentation
3. Create issue with workflow failure details
4. Contact DevOps team for infrastructure issues

## Compliance and Security

These workflows implement:
- ✅ **SLSA Level 3** build provenance
- ✅ **SPDX 2.3** SBOM generation  
- ✅ **NIST Cybersecurity Framework** alignment
- ✅ **SOC 2 Type II** compliance preparation
- ✅ **Quantum-safe cryptography** integration
- ✅ **Zero-trust security model** principles

## Performance Metrics

Expected workflow execution times:
- **CI Pipeline**: 15-25 minutes
- **Security Scan**: 10-15 minutes  
- **Performance Benchmark**: 20-30 minutes
- **Release Process**: 30-45 minutes

Optimize by:
- Using workflow caching effectively
- Parallelizing independent jobs
- Pre-building container images
- Quantum simulator warm-up strategies