from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union

import optuna
from optuna.samplers import TPESampler

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.ml_models.knowledge_tracing import KnowledgeTracingModel
from pedago_ml.ml_models.ncf import NCFModel
from pedago_ml.ml_models.xgb import XGBModel
from pedago_ml.utils.config import (
    HistoryBasedPipelineConfig,
    KnowledgeTracingModelConfig,
    ModelSearchSpaceEntry,
    ModelTuningConfig,
    NCFModelConfig,
    XGBModelConfig,
)

ModelConfig = Union[XGBModelConfig, NCFModelConfig, KnowledgeTracingModelConfig]
ModelKey = Literal["xgb", "ncf", "knowledge_tracing"]


@dataclass(frozen=True)
class TuningResult:
    """Outcome of ``run_model_tuning``."""

    study: optuna.Study
    model_config: ModelConfig
    best_metric: float


def _make_hashable(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_make_hashable(i) for i in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
    return x


def _to_jsonish(value: Any) -> Any:
    """Tuples from Optuna categoricals → lists for pydantic / model constructors."""
    if isinstance(value, tuple):
        return [_to_jsonish(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonish(v) for v in value]
    return value


def suggest_one(
    trial: optuna.Trial, param_name: str, entry: ModelSearchSpaceEntry
) -> Any:
    """Sample a single parameter from ``entry``."""
    if entry.type == "float":
        if entry.low is None or entry.high is None:
            raise ValueError(f"Float parameter '{param_name}' requires low/high")

        low, high = float(entry.low), float(entry.high)

        if entry.distribution == "loguniform":
            return trial.suggest_float(param_name, low, high, log=True)
        return trial.suggest_float(param_name, low, high, log=False)

    if entry.type == "int":
        if entry.low is None or entry.high is None:
            raise ValueError(f"Int parameter '{param_name}' requires low/high")

        low_i = int(entry.low)
        high_i = int(entry.high)

        if entry.distribution == "loguniform":
            return trial.suggest_int(param_name, low_i, high_i, log=True)
        return trial.suggest_int(param_name, low_i, high_i, log=False)

    choices = entry.choices or []
    hashable = [_make_hashable(c) for c in choices]
    picked = trial.suggest_categorical(param_name, hashable)
    return _to_jsonish(picked)


def _coerce_study_param_value(value: Any) -> Any:
    """Optuna may return tuples for categorical lists; pydantic expects lists."""
    if isinstance(value, tuple):
        return [_coerce_study_param_value(v) for v in value]
    return value


def _normalize_metric(metric: str) -> str:
    """Normalize metric keys to a consistent lookup form."""
    return metric.strip().lower().replace("-", "_")


def metric_from_eval(metrics: dict[str, float], tuning: ModelTuningConfig) -> float:
    """Extract the tuned objective value from a model's ``evaluate`` output."""
    name = tuning.metric
    normalized = {_normalize_metric(k): v for k, v in metrics.items()}
    lookup = _normalize_metric(name)

    if lookup in normalized:
        return float(normalized[lookup])

    if lookup in {"log_loss", "logloss", "cross_entropy"}:
        for alias in ("log_loss", "logloss"):
            if alias in normalized:
                return float(normalized[alias])
    raise KeyError(f"Metric '{name}' not in evaluation output {list(metrics.keys())}")


def sampler(tuning: ModelTuningConfig) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler honoring the tuning seed (if provided)."""
    if tuning.seed is not None:
        return TPESampler(seed=tuning.seed)
    return TPESampler()


def get_subconfig(
    pipeline_config: HistoryBasedPipelineConfig, model_key: ModelKey
) -> Tuple[ModelConfig, type[BaseMLModel], bool]:
    """Return (model_config, model_class, needs_val_fit) for a given model key."""
    if model_key == "xgb":
        return pipeline_config.xgb_config, XGBModel, False
    if model_key == "ncf":
        return pipeline_config.ncf_config, NCFModel, True
    if model_key == "knowledge_tracing":
        return pipeline_config.knowledge_tracing_config, KnowledgeTracingModel, False
    raise ValueError(f"Unknown model_key: {model_key}")


def merge_params(mc: ModelConfig, trial_params: dict[str, Any]) -> ModelConfig:
    """Return a copy of ``mc`` with ``model_params`` updated by trial values."""
    dumped = mc.model_params.model_dump()
    for k, v in trial_params.items():
        if k not in dumped:
            raise ValueError(
                f"Unknown tuned field '{k}' for {mc.model_name}. "
                f"Valid keys include: {sorted(dumped)!r}"
            )
        dumped[k] = _coerce_study_param_value(v)
    params_cls = type(mc.model_params)
    merged_params = params_cls(**dumped)
    return mc.model_copy(deep=True, update={"model_params": merged_params})
