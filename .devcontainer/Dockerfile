# Development Container for Quantum MLOps
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Install quantum computing system dependencies
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
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    tree \
    htop \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com | sh

# Install pip-tools for dependency management
RUN pip install --no-cache-dir pip-tools pipdeptree

# Set up development user
USER vscode
WORKDIR /workspace

# Install Python development dependencies
COPY requirements-dev.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/requirements-dev.txt

# Configure Git (will be overridden by user config)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Set up shell
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias quantum="python -m quantum_mlops.cli"' >> ~/.bashrc \
    && echo 'export PYTHONPATH=/workspace/src:$PYTHONPATH' >> ~/.bashrc

# Development prompt
RUN echo 'export PS1="[quantum-dev] \w $ "' >> ~/.bashrc