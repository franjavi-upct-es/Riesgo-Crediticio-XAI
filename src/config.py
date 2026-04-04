# src/config.py
"""Centralized configuration management using pydantic-settings.

All configurable values live here. Modules import the settings singleton
instead of hardcoding paths, ports, thresholds, or hyperparameters.

Resolution order: environment variables > .env file > defaults.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _APISettings(BaseSettings):
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = True
    log_level: str = "info"
    log_format: str = "console"  # "console" for dev, "json" for production
    api_key: str | None = None
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://localhost:5173",
        ]
    )
    rate_limit: str = "60/minute"  # Default: 60 requests per minute per client
    rate_limit_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="API_")


class _ModelSettings(BaseSettings):
    """Model artifact paths."""

    dir: Path = Path("models")
    filename: str = "xgb_model.pkl"
    feature_names_filename: str = "feature_names.pkl"

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    @property
    def model_path(self) -> Path:
        return self.dir / self.filename

    @property
    def feature_names_path(self) -> Path:
        return self.dir / self.feature_names_filename


class _DataSettings(BaseSettings):
    """Data paths and dataset configuration."""

    dir: Path = Path("data")
    synthetic_test_filename: str = "synthetic_test_set.csv"
    evaluation_metrics_filename: str = "evaluation_metrics.json"
    uci_dataset_id: int = 144

    model_config = SettingsConfigDict(env_prefix="DATA_")

    @property
    def synthetic_test_path(self) -> Path:
        return self.dir / self.synthetic_test_filename

    @property
    def evaluation_metrics_path(self) -> Path:
        return self.dir / self.evaluation_metrics_filename


class _TrainSettings(BaseSettings):
    """Training hyperparameters and configuration."""

    test_size: float = 0.2
    random_state: int = 42
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1

    model_config = SettingsConfigDict(env_prefix="TRAIN_")

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("test_size must be between 0 and 1 (exclusive)")
        return v


class _ShapSettings(BaseSettings):
    """SHAP explanation configuration."""

    significance_threshold: float = 0.001

    model_config = SettingsConfigDict(env_prefix="SHAP_")


class _MLflowSettings(BaseSettings):
    """MLflow experiment tracking configuration."""

    enabled: bool = True
    tracking_uri: str = "mlruns"  # Local directory or remote server URI
    experiment_name: str = "credit-risk-xai"
    log_models: bool = True

    model_config = SettingsConfigDict(env_prefix="MLFLOW_")


class _OtelSettings(BaseSettings):
    """OpenTelemetry distributed tracing configuration."""

    enabled: bool = False
    service_name: str = "credit-risk-api"
    exporter_endpoint: str = "http://localhost:4317"  # OTLP gRPC endpoint

    model_config = SettingsConfigDict(env_prefix="OTEL_")


class _DriftSettings(BaseSettings):
    """Data and model drift detection configuration."""

    reference_window_size: int = 200  # Number of reference samples to keep
    detection_threshold: float = 0.05  # p-value threshold for drift detection
    buffer_size: int = 50  # Minimum predictions before running drift check

    model_config = SettingsConfigDict(env_prefix="DRIFT_")


class Settings(BaseSettings):
    """Root settings object aggregating all configuration sections.

    Usage::

        from config import settings

        model_path = settings.model.model_path
        api_port = settings.api.port
    """

    api: _APISettings = Field(default_factory=_APISettings)
    model: _ModelSettings = Field(default_factory=_ModelSettings)
    data: _DataSettings = Field(default_factory=_DataSettings)
    train: _TrainSettings = Field(default_factory=_TrainSettings)
    shap: _ShapSettings = Field(default_factory=_ShapSettings)
    mlflow: _MLflowSettings = Field(default_factory=_MLflowSettings)
    otel: _OtelSettings = Field(default_factory=_OtelSettings)
    drift: _DriftSettings = Field(default_factory=_DriftSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this, not the class.
settings = Settings()
