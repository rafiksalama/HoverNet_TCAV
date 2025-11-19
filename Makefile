.PHONY: help install test test-unit test-integration test-validation test-all coverage format lint clean

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	@echo "✅ Installation complete!"

test: test-unit ## Run unit tests (default)

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ \
		-m "unit and not gpu and not data" \
		-v \
		--tb=short \
		--maxfail=5

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/integration/ \
		-m "integration and not gpu and not data" \
		-v \
		--tb=short

test-validation: ## Run validation and success criteria tests
	@echo "Running validation tests..."
	pytest tests/validation/ \
		-m "validation and not data" \
		-v \
		--tb=short

test-phase1: ## Run Phase 1 (HoverNet) tests
	@echo "Running Phase 1 tests..."
	pytest -m phase1 -v

test-phase2: ## Run Phase 2 (TCAV) tests
	@echo "Running Phase 2 tests..."
	pytest -m phase2 -v

test-phase3: ## Run Phase 3 (MIL) tests
	@echo "Running Phase 3 tests..."
	pytest -m phase3 -v

test-success: ## Run all success criteria tests
	@echo "Running success criteria tests..."
	pytest -m success_criteria -v --tb=short

test-all: ## Run all tests with coverage
	@echo "Running all tests with coverage..."
	pytest tests/ \
		-m "not gpu and not data" \
		--cov=src \
		--cov-report=html \
		--cov-report=term \
		--cov-report=xml \
		-v

test-fast: ## Run only fast tests (skip slow)
	@echo "Running fast tests..."
	pytest tests/ \
		-m "not slow and not gpu and not data" \
		-v

coverage: test-all ## Generate coverage report
	@echo "Opening coverage report..."
	@command -v open >/dev/null 2>&1 && open htmlcov/index.html || echo "Coverage report: htmlcov/index.html"

format: ## Format code with black and isort
	@echo "Formatting code..."
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black
	@echo "✅ Code formatted!"

lint: ## Run linters (flake8, pylint)
	@echo "Running linters..."
	@echo "→ flake8..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "→ pylint..."
	pylint src/ --fail-under=7.0
	@echo "✅ Linting passed!"

typecheck: ## Run type checker (mypy)
	@echo "Running type checker..."
	mypy src/ --ignore-missing-imports
	@echo "✅ Type checking passed!"

check: format lint typecheck test-all ## Run all checks (format, lint, type, test)
	@echo "✅ All checks passed!"

clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

docs: ## Build documentation
	@echo "Building documentation..."
	cd docs && make html
	@echo "✅ Documentation built!"

jupyter: ## Start Jupyter Lab
	@echo "Starting Jupyter Lab..."
	jupyter lab --notebook-dir=notebooks/

setup-dev: install ## Set up development environment
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "✅ Development environment ready!"

verify: ## Verify installation
	@echo "Verifying installation..."
	@echo "→ Python version:"
	python --version
	@echo "→ PyTorch:"
	python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
	@echo "→ TensorFlow:"
	python -c "import tensorflow as tf; print(f'  Version: {tf.__version__}')"
	@echo "→ OpenSlide:"
	python -c "import openslide; print(f'  Version: {openslide.__version__}')"
	@echo "→ Running basic tests..."
	pytest tests/unit/ -m "unit and not gpu and not data" --maxfail=1 -q
	@echo "✅ Installation verified!"

download-data: ## Instructions for downloading datasets
	@echo "📥 Dataset Download Instructions:"
	@echo ""
	@echo "1. HER2-TUMOR-ROIS (Primary dataset):"
	@echo "   URL: https://www.cancerimagingarchive.net/collection/her2-tumor-rois/"
	@echo "   - Register at TCIA (free)"
	@echo "   - Download NBIA Data Retriever"
	@echo "   - Add collection to cart and download"
	@echo ""
	@echo "2. Post-NAT-BRCA (Secondary dataset):"
	@echo "   URL: https://www.cancerimagingarchive.net/collection/post-nat-brca/"
	@echo ""
	@echo "See QUICK_START_GUIDE.md for detailed instructions"

success-report: ## Generate success criteria report
	@echo "📊 Success Criteria Status:"
	@echo ""
	@echo "Phase 1: HoverNet Segmentation"
	@pytest tests/validation/test_success_criteria.py::TestPhase1Success --collect-only -q 2>/dev/null | head -10 || echo "  Tests not yet run"
	@echo ""
	@echo "Phase 2: TCAV Integration"
	@pytest tests/validation/test_success_criteria.py::TestPhase2Success --collect-only -q 2>/dev/null | head -10 || echo "  Tests not yet run"
	@echo ""
	@echo "Phase 3: MIL Model"
	@pytest tests/validation/test_success_criteria.py::TestPhase3Success --collect-only -q 2>/dev/null | head -10 || echo "  Tests not yet run"
	@echo ""
	@echo "Phase 4: Interpretability"
	@pytest tests/validation/test_success_criteria.py::TestPhase4Success --collect-only -q 2>/dev/null | head -10 || echo "  Tests not yet run"

watch: ## Watch for changes and run tests automatically
	@echo "Watching for changes..."
	pytest-watch tests/ -m "not slow and not gpu and not data"
