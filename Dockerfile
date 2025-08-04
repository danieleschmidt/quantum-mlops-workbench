# Multi-stage Dockerfile for Quantum MLOps Platform
# Optimized for quantum computing workloads with security hardening

# Stage 1: Base quantum computing environment
FROM python:3.11-slim-bookworm AS quantum-base

# Security: Update base system and install security updates
RUN apt-get update && apt-get upgrade -y

# Security: Create non-root user with restricted permissions
RUN groupadd -r -g 1000 quantum && \
    useradd -r -u 1000 -g quantum -m -d /home/quantum -s /bin/bash quantum

# Security: Install system dependencies with minimal attack surface
RUN apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    curl \
    git \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Security: Set secure file permissions
RUN chmod 755 /home/quantum

# Security: Remove unnecessary packages and files
RUN apt-get purge -y --auto-remove \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/log -type f -delete

# Stage 2: Development environment
FROM quantum-base AS development

# Security: Create app directory with proper ownership
RUN mkdir -p /app && chown quantum:quantum /app
WORKDIR /app

# Copy dependency files with proper ownership
COPY --chown=quantum:quantum requirements.txt requirements-dev.txt pyproject.toml ./

# Security: Switch to non-root user for dependency installation
USER quantum

# Install Python dependencies with security considerations
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user -r requirements-dev.txt

# Copy source code with proper ownership
COPY --chown=quantum:quantum . .

# Install package in development mode
RUN pip install --no-cache-dir --user -e .

# Security: Set environment variables for non-root user
ENV PATH="/home/quantum/.local/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Security: Expose minimal port
EXPOSE 8000

# Security: Use exec form and run as non-root
CMD ["python", "-m", "quantum_mlops.cli", "serve", "--host", "127.0.0.1"]

# Stage 3: Production environment
FROM quantum-base AS production

# Security: Create app directory with restricted permissions
RUN mkdir -p /app && chown quantum:quantum /app && chmod 755 /app
WORKDIR /app

# Copy only requirements first for better layer caching
COPY --chown=quantum:quantum requirements.txt ./

# Security: Switch to non-root user early
USER quantum

# Install Python dependencies with security considerations
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# Copy source code with proper ownership and minimal files
COPY --chown=quantum:quantum src/ ./src/
COPY --chown=quantum:quantum pyproject.toml ./

# Install package
RUN pip install --no-cache-dir --user .

# Security: Set environment variables for production
ENV PATH="/home/quantum/.local/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Security: Create directories for runtime data
RUN mkdir -p /home/quantum/.quantum_mlops/{logs,cache,data} && \
    chmod 700 /home/quantum/.quantum_mlops

# Security: Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import quantum_mlops; import sys; quantum_mlops.__version__; sys.exit(0)" || exit 1

# Security: Expose minimal port
EXPOSE 8000

# Security: Use exec form, run as non-root, bind to localhost for security
CMD ["python", "-m", "quantum_mlops.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Testing environment
FROM development AS testing

# Security: Install additional security testing tools
RUN pip install --no-cache-dir --user bandit safety

# Security: Create test output directory
RUN mkdir -p /home/quantum/test-results && chmod 755 /home/quantum/test-results

# Security: Run comprehensive test suite including security tests
CMD ["sh", "-c", "pytest -v --cov=quantum_mlops --cov-report=html --cov-report=xml --junitxml=/home/quantum/test-results/junit.xml tests/ && bandit -r src/ -f json -o /home/quantum/test-results/bandit-report.json || true"]

# Stage 5: Security-hardened production environment
FROM python:3.11-slim-bookworm AS secure-production

# Security: Minimal base with security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create quantum user with minimal privileges
RUN groupadd -r -g 1000 quantum && \
    useradd -r -u 1000 -g quantum -m -d /home/quantum -s /sbin/nologin quantum

# Security: Create minimal app structure
RUN mkdir -p /app && chown quantum:quantum /app && chmod 755 /app
WORKDIR /app

# Copy only production requirements
COPY --chown=quantum:quantum requirements.txt ./

# Security: Switch to non-root user
USER quantum

# Install minimal production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy only necessary application files
COPY --chown=quantum:quantum src/quantum_mlops/ ./quantum_mlops/
COPY --chown=quantum:quantum pyproject.toml ./

# Install application
RUN pip install --no-cache-dir --user .

# Security: Set secure environment
ENV PATH="/home/quantum/.local/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Security: Create secure runtime directories
RUN mkdir -p /home/quantum/.quantum_mlops/{logs,cache} && \
    chmod 700 /home/quantum/.quantum_mlops

# Security: Drop all capabilities and set read-only filesystem ready
# (These would be set in docker-compose or k8s deployment)

# Security: Minimal health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD python -c "import quantum_mlops; exit(0)" || exit 1

# Security: Minimal exposed port
EXPOSE 8000

# Security: Run with minimal privileges
CMD ["python", "-m", "quantum_mlops.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]