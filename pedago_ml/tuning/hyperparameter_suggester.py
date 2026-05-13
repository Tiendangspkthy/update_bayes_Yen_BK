from typing import Any, Mapping

import optuna

from pedago_ml.utils.config import ModelSearchSpaceEntry
from pedago_ml.utils.logger import logger


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


class HyperparameterSuggester:
    """
    Suggests hyperparameters from a mapping of parameter name → ModelSearchSpaceEntry.
    """

    def suggest_all(
        self,
        trial: optuna.Trial,
        search_space: Mapping[str, ModelSearchSpaceEntry],
    ) -> dict[str, Any]:
        """Suggest a full dict of param_name -> sampled value for one Optuna trial."""
        suggested: dict[str, Any] = {}
        for name, entry in search_space.items():
            try:
                suggested[name] = suggest_one(trial, name, entry)
            except Exception as e:
                logger.error("Failed to suggest parameter '{}': {}", name, e)
                raise
        return suggested
