# Credit Risk XAI

> Production-grade credit risk classification with SHAP explanations

## Overview

End-to-end MLOps pipeline for credit risk classification using XGBoost + SHAP. Designed as a production-ready system with proper architecture, security, observability, experiment tracking, drift detection, CI/CD, and Kubernetes deployment — even without a live deployment target.

**System layers:**

1. **Training pipeline** — config-driven (YAML), with MLflow experiment tracking, evaluation metrics, and artifact versioning.
2. **FastAPI API** — predictions with SHAP explanations, evaluation endpoints, API key auth, rate limiting, Prometheus metrics, structured JSON logging, drift detection, and OpenTelemetry tracing.
3. **React + TypeScript dashboard** — three-page SPA: global evaluation, interactive local prediction, monitoring overview.
4. **Observability** — Prometheus + Grafana + AlertManager + Jaeger. 11 custom metrics, 8 alerting rules, distributed tracing.
5. **CI/CD + K8s** — GitHub Actions (lint → test → build → push → deploy), Docker images, production Kubernetes manifests.

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
make install-dev

# 2. Train (model + evaluation metrics + MLflow run)
make train

# 3. Run API + dashboard
make api          # Terminal 1 — http://localhost:8000/docs
make dashboard    # Terminal 2 — http://localhost:5173
```

### Docker (full observability stack)

```bash
make train && make docker-up

# Dashboard:     http://localhost:5173
# API docs:      http://localhost:8000/docs
# Prometheus:    http://localhost:9090
# AlertManager:  http://localhost:9093
# Grafana:       http://localhost:3000 (admin/admin)
# Jaeger UI:     http://localhost:16686
```

## Project Structure

```
credit-risk-xai/
├── pyproject.toml                          # Package metadata and tool config
├── Makefile                                # Development task runner
├── configs/training.yml                    # Training hyperparameters (YAML)
├── Dockerfile.api / Dockerfile.dashboard   # Multi-stage production images
├── docker-compose.yml                      # 6 services: API, Dashboard, Prometheus,
│                                           #   AlertManager, Grafana, Jaeger
├── src/
│   ├── config.py                           # Centralized settings (8 sections)
│   ├── data/
│   │   ├── loader.py                       # UCI dataset fetching
│   │   └── preprocessing.py                # Shared encoding (train + serve)
│   ├── model/
│   │   ├── train.py                        # Training pipeline (YAML + MLflow)
│   │   ├── training_config.py              # YAML config loader + typed dataclasses
│   │   ├── experiment_tracker.py           # MLflow context manager
│   │   ├── evaluate.py                     # Metrics computation + JSON persistence
│   │   └── registry.py                     # Artifact loading and validation
│   ├── explain/
│   │   └── shap_engine.py                  # SHAP computation and normalization
│   ├── api/
│   │   ├── app.py                          # FastAPI factory + OTel + lifespan
│   │   ├── auth.py                         # API key auth (constant-time)
│   │   ├── schemas.py                      # Request/response models
│   │   ├── dependencies.py                 # DI (model, explainer, drift detector)
│   │   ├── middleware.py                   # CORS, security, metrics, rate limiting
│   │   └── routes/
│   │       ├── health.py                   # /health, /alive
│   │       ├── predict.py                  # /predict_risk/ (metrics + drift + OTel)
│   │       ├── evaluation.py               # /evaluation/* (6 endpoints)
│   │       └── drift.py                    # /monitoring/drift (GET + POST analyze)
│   └── monitoring/
│       ├── metrics.py                      # Prometheus metrics + /metrics endpoint
│       ├── drift.py                        # KS-test drift detection + Prometheus gauges
│       └── tracing.py                      # OpenTelemetry setup + auto-instrumentation
│
├── dashboard/                              # React + TypeScript + Tailwind (Vite)
│   └── src/ (21 files)
│
├── tests/ (157 tests, 79% coverage)
│   ├── unit/ (12 test files)
│   └── integration/ (2 test files)
│
├── .github/workflows/
│   ├── ci.yml                              # Lint → test → build (push/PR)
│   └── cd.yml                              # Build → push GHCR → deploy K8s (tags)
│
└── infra/
    ├── k8s/ (7 manifests)                  # Namespace, ConfigMap, Secret, Deployment,
    │                                       #   Service, Ingress, HPA
    ├── prometheus/
    │   ├── prometheus.yml                  # Scrape config + alertmanager target
    │   └── alerts.yml                      # 8 alerting rules (3 groups)
    ├── alertmanager/
    │   └── alertmanager.yml                # Severity routing + inhibition
    └── grafana/provisioning/
```

## Drift Detection

The API detects data and prediction drift using statistical tests (Kolmogorov-Smirnov) comparing live prediction inputs against the training reference distribution.

Every call to `/predict_risk/` records the preprocessed input vector and predicted probability in a rolling buffer. When the buffer reaches `DRIFT_BUFFER_SIZE` (default: 50), drift can be analyzed:

```bash
# Check current drift status
curl http://localhost:8000/monitoring/drift

# Trigger manual analysis
curl -X POST http://localhost:8000/monitoring/drift/analyze
```

Three Prometheus gauges are updated on each analysis: `drift_score` (fraction of features drifted), `drift_features_drifted_count`, and `prediction_drift_pvalue`. These feed the alerting rules in `alerts.yml`.

## OpenTelemetry Tracing

When `OTEL_ENABLED=true`, the API auto-instruments all FastAPI requests and adds manual spans for preprocessing, model inference, and SHAP computation. Traces are exported via OTLP gRPC to a collector (Jaeger in docker-compose).

View traces at `http://localhost:16686` (Jaeger UI). Search for service `credit-risk-api`.

## Alerting

8 Prometheus alerting rules across 3 groups:

| Group | Alert | Condition | Severity |
|-------|-------|-----------|----------|
| API health | HighErrorRate | 5xx rate > 5% for 2m | critical |
| API health | HighLatencyP99 | p99 > 2s for 3m | warning |
| API health | HighLatencyP50 | p50 > 500ms for 5m | warning |
| API health | APIDown | scrape target down for 1m | critical |
| Predictions | PredictionErrorSpike | errors > 0.1/s for 2m | critical |
| Predictions | SHAPComputationSlow | p95 > 5s for 3m | warning |
| Predictions | HighRiskPredictionSpike | > 70% high-risk for 10m | warning |
| Drift | DataDriftDetected | drift_score > 30% for 5m | warning |
| Drift | SevereDataDrift | drift_score > 50% for 5m | critical |
| Drift | PredictionDistributionDrift | KS p-value < 0.01 for 5m | warning |

AlertManager routes critical alerts with 10s group wait (1h repeat) and warnings with 1m group wait (4h repeat). Critical alerts inhibit warnings for the same alert name.

## Development

```bash
make help              # All commands
make test              # 157 tests with coverage
make lint              # Ruff + tsc
make check             # lint + type-check + tests
make docker-up/down    # Full 6-service stack
make clean             # Remove generated files
```

## Roadmap

- [x] Phase 1: Package structure, shared preprocessing, config, tests
- [x] Phase 2: Auth, rate limiting, Prometheus metrics, structured logging, Docker
- [x] Phase 3: React + TypeScript dashboard, evaluation API
- [x] Phase 4: MLflow, YAML config, GitHub Actions CI/CD, Kubernetes manifests
- [x] Phase 5: Drift detection, alerting rules, OpenTelemetry tracing
