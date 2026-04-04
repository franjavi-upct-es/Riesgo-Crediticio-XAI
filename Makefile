# credit-risk-xai/Makefile
#
# Common development tasks. Run `make help` for a summary.

.PHONY: help install install-dev train api dashboard dashboard-build test lint type-check fmt check docker-up docker-down clean

PYTHON ?= uv run python
PIP    ?= uv pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Install the package (production dependencies only)
	$(PIP) install -e .

install-dev: ## Install with all extras (dev, dashboard, monitoring)
	$(PIP) install -e ".[all]"

install-dashboard: ## Install dashboard dependencies
	cd dashboard && npm install

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------

train: ## Run the training pipeline (model + evaluation metrics)
	$(PYTHON) -m src.model.train

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

api: ## Start the API server (port 8000)
	$(PYTHON) -m src.api.app

dashboard: ## Start the dashboard dev server (port 5173)
	cd dashboard && npx vite

dashboard-build: ## Build the dashboard for production
	cd dashboard && npm run build

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Run all tests with coverage
	$(PYTHON) -m pytest tests/ --cov=credit_risk --cov-report=term-missing --cov-report=html -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -v

lint: ## Run ruff linter (Python) + tsc (TypeScript)
	$(PYTHON) -m ruff check src/ tests/
	cd dashboard && npx tsc --noEmit

lint-fix: ## Run ruff with auto-fix
	$(PYTHON) -m ruff check src/ tests/ --fix

type-check: ## Run mypy type checker
	$(PYTHON) -m mypy src/

fmt: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/

check: lint type-check test ## Run all quality checks

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-up: ## Start full stack (API + Dashboard + Prometheus + Grafana)
	docker compose up --build -d

docker-down: ## Stop and remove containers
	docker compose down -v

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove generated files and caches
	rm -rf models/ data/ .pytest_cache/ htmlcov/ .mypy_cache/ .ruff_cache/
	rm -rf dashboard/dist/ dashboard/node_modules/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
