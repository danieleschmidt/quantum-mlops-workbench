.PHONY: help install install-dev test test-quantum lint format type-check clean docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Quantum MLOps Workbench - Development Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run all tests
	pytest tests/ -v --cov=quantum_mlops --cov-report=term-missing

test-quantum: ## Run quantum-specific tests
	pytest tests/quantum/ -v -m "quantum or simulation"

test-fast: ## Run fast tests only (skip slow tests)
	pytest tests/ -v -m "not slow"

lint: ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/
	isort src/ tests/

type-check: ## Run type checking
	mypy src/

security: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	python -m build

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	python -m twine upload dist/*

quantum-status: ## Check quantum backend status
	quantum-mlops status

quantum-test: ## Run quantum hardware tests
	quantum-mlops test --backend simulator --shots 1000