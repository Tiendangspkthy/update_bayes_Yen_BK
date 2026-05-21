import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, model_validator

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


class MLModelOutputConfig(BaseModel):
    score_col: str = "pred_proba"
    prediction_col: str = "prediction"
    rank_col: str = "rank"


class ModelSearchSpaceEntry(BaseModel):
    type: Literal["float", "int", "categorical"]
    distribution: Literal["uniform", "loguniform"] = "uniform"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def _validate_by_type(self) -> "ModelSearchSpaceEntry":
        if self.type in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Search space '{self.type}' parameters require " "'low' and 'high'"
                )
            if self.low >= self.high:
                raise ValueError("'low' must be < 'high'")
            if self.distribution == "loguniform" and self.low <= 0:
                raise ValueError("'loguniform' requires low > 0")
            if self.choices is not None:
                raise ValueError("Numeric search space entries must not set 'choices'")
        else:
            if not self.choices:
                raise ValueError("categorical entries require non-empty 'choices'")
            if self.low is not None or self.high is not None:
                raise ValueError("categorical entries must not set 'low'/'high'")
        return self


class ModelTuningConfig(BaseModel):
    enabled: bool = False
    n_trials: int = 20
    study_name: str = "hyperparameter_tuning"
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "logloss"
    seed: Optional[int] = None
    search_space: Dict[str, ModelSearchSpaceEntry] = Field(default_factory=dict)


class XGBParamsConfig(BaseModel):
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


class XGBModelConfig(BaseConfig):
    model_name: str = "xgb"

    threshold: float = 0.5
    feature_columns: Optional[list[str]] = None

    model_params: XGBParamsConfig = XGBParamsConfig()
    tuning: Optional[ModelTuningConfig] = None
    output: MLModelOutputConfig = MLModelOutputConfig()


class NCFParamsConfig(BaseModel):
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


class NCFModelConfig(BaseConfig):
    model_name: str = "ncf"

    threshold: float = 0.5
    feature_columns: Optional[list[str]] = None

    model_params: NCFParamsConfig = NCFParamsConfig()
    tuning: Optional[ModelTuningConfig] = None
    output: MLModelOutputConfig = MLModelOutputConfig()


class KnowledgeTracingParamsConfig(BaseModel):
    alpha: float = 1.0
    beta: float = 1.0

    w0: float = 0.0
    w1: float = 3.0
    w2: float = 2.0

    category_columns: list[str] = Field(
        default_factory=lambda: [
            "a_category",
            "b_category",
            "c_category",
            "d_category",
            "e_category",
        ]
    )
    category_aggregation: Literal["mean", "max"] = "mean"
    item_accuracy_col: str = "item_accuracy"

    clip_probability: bool = True


class KnowledgeTracingModelConfig(BaseConfig):
    model_name: str = "knowledge_tracing"
    threshold: float = 0.5
    probability_col: Optional[str] = None
    model_params: KnowledgeTracingParamsConfig = KnowledgeTracingParamsConfig()
    tuning: Optional[ModelTuningConfig] = None
    output: MLModelOutputConfig = MLModelOutputConfig()


class ZPDParamsConfig(BaseModel):
    score_mode: Literal["probability", "ability_difficulty"] = "probability"

    lower_probability: float = 0.55
    target_probability: float = 0.65
    upper_probability: float = 0.75

    lower_difficulty_gap: float = -0.5
    target_difficulty_gap: float = 0.0
    upper_difficulty_gap: float = 0.5

    min_in_zone_score: float = 0.7
    outside_tolerance: float = 0.25
    clip_probability: bool = True

    base_score_weight: float = 0.4
    zpd_score_weight: float = 0.6


class ZPDOutputConfig(BaseModel):
    zpd_value_col: str = "zpd_value"
    zpd_distance_col: str = "zpd_distance"
    zpd_score_col: str = "zpd_score"
    zpd_label_col: str = "zpd_label"
    is_aligned_col: str = "is_zpd_aligned"
    adjusted_score_col: str = "zpd_adjusted_score"
    rank_col: str = "zpd_rank"


class ZPDTheoryConfig(BaseConfig):
    model_name: str = "zpd"
    probability_col: Optional[str] = "xgb_pred_proba"

    base_score_col: Optional[str] = None

    ability_col: Optional[str] = None
    item_difficulty_col: Optional[str] = None

    model_params: ZPDParamsConfig = ZPDParamsConfig()
    output: ZPDOutputConfig = ZPDOutputConfig()


class FlowParamsConfig(BaseModel):
    lower_probability: float = 0.45
    target_probability: float = 0.60
    upper_probability: float = 0.75
    min_in_flow_score: float = 0.7
    outside_tolerance: float = 0.25
    clip_probability: bool = True
    base_score_weight: float = 0.5
    flow_score_weight: float = 0.5


class FlowOutputConfig(BaseModel):
    flow_value_col: str = "flow_value"
    flow_distance_col: str = "flow_distance"
    flow_score_col: str = "flow_score"
    flow_label_col: str = "flow_label"
    is_aligned_col: str = "is_flow_aligned"
    adjusted_score_col: str = "flow_adjusted_score"
    rank_col: str = "flow_rank"


class FlowTheoryConfig(BaseConfig):
    model_name: str = "flow_theory"
    probability_col: Optional[str] = "xgb_pred_proba"
    base_score_col: Optional[str] = None
    model_params: FlowParamsConfig = FlowParamsConfig()
    output: FlowOutputConfig = FlowOutputConfig()


class BasePipelineConfig(BaseConfig):
    interaction_data_path: str
    item_data_path: str
    user_id: str
    item_id: str
    target: str
    item_features: list[ItemFeatureConfig]

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        data = resolve_dict(data)
        instance = cls(**data)
        instance.load_from_env()
        return instance


class HistoryBasedPipelineConfig(BasePipelineConfig):
    splitter: SplitterConfig
    smoothing: int
    xgb_config: XGBModelConfig = XGBModelConfig()
    ncf_config: NCFModelConfig = NCFModelConfig()
    zpd_config: ZPDTheoryConfig = ZPDTheoryConfig()
    flow_config: FlowTheoryConfig = FlowTheoryConfig()
    knowledge_tracing_config: KnowledgeTracingModelConfig = (
        KnowledgeTracingModelConfig()
    )


class RaschParamsConfig(BaseModel):
    max_iter: int = 100
    tol: float = 1e-4
    difficulty_prior_std: float = 2.0
    clip_difficulty: float = 6.0


class RaschOutputConfig(BaseModel):
    score_col: str = "rasch_pred_proba"
    prediction_col: str = "rasch_prediction"
    rank_col: str = "rasch_rank"
    difficulty_col: str = "item_difficulty"


class RaschModelConfig(BaseConfig):
    model_name: str = "rasch"
    threshold: float = 0.5
    ability_col: str = "ability"
    model_params: RaschParamsConfig = RaschParamsConfig()
    tuning: Optional[ModelTuningConfig] = None
    output: RaschOutputConfig = RaschOutputConfig()


class GridPosteriorConfig(BaseModel):
    theta_min: float = -4.0
    theta_max: float = 4.0
    n_grid: int = 81
    prior_mean: float = 0.0
    prior_std: float = 1.0
    estimator: Literal["mean", "map"] = "mean"


class SimilarityPenaltyConfig(BaseModel):
    topic_weight: float = 0.5
    category_weight: float = 0.5
    recent_window: int = 3
    lambda_penalty: float = 0.3


class ColdStartZPDConfig(BaseModel):
    """Hard ZPD band on expected response probability P(correct | posterior)."""

    lower_probability: float = 0.55
    target_probability: float = 0.65
    upper_probability: float = 0.75

    @model_validator(mode="after")
    def validate_bounds(self) -> "ColdStartZPDConfig":
        lower = float(self.lower_probability)
        target = float(self.target_probability)
        upper = float(self.upper_probability)
        if not (lower <= target <= upper):
            raise ValueError(
                "ZPD bounds must satisfy lower_probability <= "
                "target_probability <= upper_probability"
            )
        return self


class ColdStartSelectorConfig(BaseModel):
    info_weight: float = 0.5
    zpd: ColdStartZPDConfig = ColdStartZPDConfig()
    similarity: SimilarityPenaltyConfig = SimilarityPenaltyConfig()


class StoppingConfig(BaseModel):
    min_questions: int = 3
    max_questions: int = 7
    variance_threshold: float = 0.25

    @model_validator(mode="after")
    def validate_thresholds(self) -> "StoppingConfig":
        if self.min_questions < 1:
            raise ValueError("min_questions must be >= 1")
        if self.max_questions < self.min_questions:
            raise ValueError("max_questions must be >= min_questions")
        if self.variance_threshold <= 0:
            raise ValueError("variance_threshold must be > 0")
        return self


class ColdStartSplitterConfig(BaseModel):
    calibration_ratio: float = 0.8
    eval_ratio: float = 0.2
    seed: int = 12

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> "ColdStartSplitterConfig":
        total = float(self.calibration_ratio + self.eval_ratio)
        if abs(total - 1.0) > 1e-8:
            raise ValueError("calibration_ratio + eval_ratio must equal 1.0")
        return self


class ColdStartPipelineConfig(BasePipelineConfig):
    cold_start_splitter: ColdStartSplitterConfig = ColdStartSplitterConfig()
    rasch_config: RaschModelConfig = RaschModelConfig()
    posterior_config: GridPosteriorConfig = GridPosteriorConfig()
    selector_config: ColdStartSelectorConfig = ColdStartSelectorConfig()
    stopping_config: StoppingConfig = StoppingConfig()


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
