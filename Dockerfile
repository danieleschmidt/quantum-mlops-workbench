# Multi-stage Dockerfile for Quantum MLOps Platform
# Optimized for quantum computing workloads with security hardening

# Stage 1: Base quantum computing environment
FROM python:3.11-slim-bookworm AS quantum-base

# Security: Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Install system dependencies for quantum computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 2: Development environment
FROM quantum-base AS development

WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER quantum

EXPOSE 8000
CMD ["python", "-m", "quantum_mlops.cli", "serve"]

# Stage 3: Production environment
FROM quantum-base AS production

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install package
RUN pip install --no-cache-dir .

# Security: Switch to non-root user
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import quantum_mlops; print('healthy')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "quantum_mlops.cli", "serve", "--host", "0.0.0.0"]

# Stage 4: Testing environment
FROM development AS testing

# Run tests by default
CMD ["pytest", "-v", "--cov=quantum_mlops", "--cov-report=html"]