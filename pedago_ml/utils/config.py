import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field

ENV_PATTERN = re.compile(r"\$\{(\w+)\}")

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """
    Base configuration factory providing layered loading.
    """

    default_config_path: str | Path | None = None

    def load_from_env(self) -> None:
        """Override config values from environment variables."""
        pass

    @classmethod
    def load_config(cls: Type[T], config_path: Optional[Union[str, Path]] = None) -> T:
        if config_path is not None:
            return cls.from_file(config_path)

        if cls.default_config_path:
            default_path = Path(cls.default_config_path)
            if default_path.exists():
                return cls.from_file(default_path)

        return cls()

    @classmethod
    def from_file(cls: Type[T], config_path: Union[str, Path]) -> T:
        config_path = Path(config_path)
        data = load_data_from_file(config_path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ItemFeatureConfig(BaseModel):
    name: str
    type: str


class SplitterConfig(BaseModel):
    train_ratio: float
    val_ratio: float
    test_ratio: float


class XGBParamsConfig(BaseModel):
    """XGBoost hyperparameters."""

    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    random_state: int = 12
    n_jobs: int = -1


class XGBOutputConfig(BaseModel):
    """XGBoost output columns and artifact path."""

    score_col: str = "xgb_pred_proba"
    prediction_col: str = "xgb_pred"
    rank_col: str = "xgb_rank"


class XGBModelConfig(BaseConfig):
    model_name: str = "xgb"

    user_id: str = "StudentID"
    item_id: str = "QuestionID"
    target_col: str = "Result"

    threshold: float = 0.5
    feature_columns: Optional[list[str]] = None

    model_params: XGBParamsConfig = XGBParamsConfig()
    output: XGBOutputConfig = XGBOutputConfig()


class NCFParamsConfig(BaseModel):
    """Neural Collaborative Filtering hyperparameters."""

    embedding_dim: int = 16
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.1

    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 7
    log_every_n_epochs: int = 5

    random_state: int = 12
    device: str = "auto"


class NCFOutputConfig(BaseModel):
    """NCF output columns."""

    score_col: str = "ncf_pred_proba"
    prediction_col: str = "ncf_pred"
    rank_col: str = "ncf_rank"


class NCFModelConfig(BaseConfig):
    """Configuration for history-based NCF correctness prediction."""

    model_name: str = "ncf"

    user_id: str = "StudentID"
    item_id: str = "QuestionID"
    target_col: str = "Result"

    threshold: float = 0.5
    feature_columns: Optional[list[str]] = None

    model_params: NCFParamsConfig = NCFParamsConfig()
    output: NCFOutputConfig = NCFOutputConfig()


class BasePipelineConfig(BaseConfig):
    interaction_data_path: str
    item_data_path: str
    user_id: str
    item_id: str
    target: str
    item_features: list[ItemFeatureConfig]
    splitter: SplitterConfig
    smoothing: int

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        data = resolve_dict(data)
        instance = cls(**data)
        instance.load_from_env()
        return instance


class HistoryBasedPipelineConfig(BasePipelineConfig):
    xgb_config: XGBModelConfig = XGBModelConfig()
    ncf_config: NCFModelConfig = NCFModelConfig()


def load_data_from_file(config_path: Path) -> Dict[str, Any]:
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        if suffix == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported format: {suffix}")


def resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        match = ENV_PATTERN.fullmatch(value)
        if match:
            env_key = match.group(1)
            env_val = os.getenv(env_key)
            if env_val is None:
                raise RuntimeError(f"Missing env var: {env_key}")
            return env_val
    return value


def resolve_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: resolve_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [resolve_dict(i) for i in d]
    return resolve_env(d)
