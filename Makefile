# credit-risk-xai/Makefile
#
# Common development tasks. Run `make help` for a summary.

.PHONY: help install install-dev train api dashboard dashboard-build test lint type-check fmt check docker-up docker-down clean

PYTHON ?= uv run python3
UV     ?= uv

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Install the package (production dependencies only)
	$(UV) sync --no-dev

install-dev: ## Install with all extras (dev, dashboard, monitoring)
	$(UV) sync --all-extras

install-dashboard: ## Install dashboard dependencies
	cd dashboard && npm install

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------

train: ## Run the training pipeline (model + evaluation metrics)
	$(UV) run python -m src.model.train

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

api: ## Start the API server (port 8000)
	$(UV) run python -m src.api.app

dashboard: ## Start the dashboard dev server (port 5173)
	cd dashboard && npx vite

dashboard-build: ## Build the dashboard for production
	cd dashboard && npm run build

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Run all tests with coverage
	$(UV) run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	$(UV) run pytest tests/integration/ -v

lint: ## Run ruff linter (Python) + tsc (TypeScript)
	$(UV) run ruff check src/ tests/
	cd dashboard && npx tsc --noEmit

lint-fix: ## Run ruff with auto-fix
	$(UV) run ruff check src/ tests/ --fix

type-check: ## Run mypy type checker
	$(UV) run mypy src/

fmt: ## Format code with ruff
	$(UV) run ruff format src/ tests/

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
