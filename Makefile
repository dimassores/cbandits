.PHONY: help install install-dev test test-cov lint format clean build check-examples

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src/cbandits --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	flake8 src/ tests/ examples/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/cbandits/ --ignore-missing-imports

format: ## Format code with black and isort
	black src/ tests/ examples/ --line-length=88
	isort src/ tests/ examples/ --profile=black --line-length=88

format-check: ## Check if code is properly formatted
	black --check --diff src/ tests/ examples/ --line-length=88
	isort --check-only --diff src/ tests/ examples/ --profile=black --line-length=88

check-examples: ## Run all simple examples to ensure they work
	python examples/simple_examples/simple_ucb_b1_example.py
	python examples/simple_examples/simple_ucb_b2_example.py
	python examples/simple_examples/simple_ucb_b2c_example.py
	python examples/simple_examples/simple_ucb_m1_example.py

check-imports: ## Test that all imports work correctly
	python -c "from cbandits import UCB_B1, UCB_B2, UCB_B2C, UCB_M1, GeneralCostRewardEnvironment; print('All imports successful')"

pre-commit: format lint test check-examples check-imports ## Run all pre-commit checks

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

check-build: ## Check the built package
	twine check dist/*

all: clean install-dev pre-commit build check-build ## Run complete development pipeline 