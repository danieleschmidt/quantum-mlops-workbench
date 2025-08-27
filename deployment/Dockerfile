# Multi-stage production Dockerfile for Quantum Meta-Learning
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY quantum_meta_learning_*.py ./
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache

# Set permissions
RUN chown -R quantum:quantum /app
USER quantum

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV QUANTUM_ENV=production
ENV QUANTUM_LOG_LEVEL=INFO

# Run application
CMD ["python", "-m", "src.quantum_mlops.api", "--host", "0.0.0.0", "--port", "8080"]
