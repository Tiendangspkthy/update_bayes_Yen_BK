import math
from typing import Any, Literal, Mapping, Optional, Union, cast

import optuna
import pandas as pd

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.base.tuning import (
    TuningResult,
    get_subconfig,
    merge_params,
    metric_from_eval,
    sampler,
    suggest_one,
)
from pedago_ml.edu_theory.zpd import ZPDTheory
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


def suggest_all(
    trial: optuna.Trial,
    search_space: Mapping[str, ModelSearchSpaceEntry],
) -> dict[str, Any]:
    """Suggest a full dict of param_name -> sampled value for one Optuna trial."""
    suggested: dict[str, Any] = {}
    for name, entry in search_space.items():
        suggested[name] = suggest_one(trial, name, entry)
    return suggested


def run_model_tuning(
    pipeline_config: HistoryBasedPipelineConfig,
    model_key: ModelKey,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tuning_override: Optional[ModelTuningConfig] = None,
) -> TuningResult:
    """
    Optuna search over ``model_config.tuning.search_space``.

    If ``tuning_override`` is set, it replaces the block from YAML for this run.
    """
    base_mc, model_cls, needs_val_fit = get_subconfig(pipeline_config, model_key)
    tuning = tuning_override or base_mc.tuning
    if tuning is None or not tuning.enabled:
        raise ValueError(
            f"Tuning is disabled or missing for {model_key}. "
            "Set tuning.enabled=true in YAML or pass tuning_override."
        )
    if not tuning.search_space:
        raise ValueError("tuning.search_space must not be empty")

    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_all(trial=trial, search_space=tuning.search_space)
        model_config_trial = merge_params(base_mc, trial_params)
        model = model_cls(config=pipeline_config, model_config=model_config_trial)
        if needs_val_fit:
            if not isinstance(model, NCFModel):
                raise TypeError(
                    "needs_val_fit=True but model is not an NCFModel. "
                    f"Got: {type(model).__name__}"
                )
            model.fit(train_df, val_df)  # pylint: disable=too-many-function-args
        else:
            model.fit(train_df)
        metrics = model.evaluate(val_df)
        value = metric_from_eval(metrics, tuning)
        if math.isnan(value):
            raise optuna.TrialPruned()
        return value

    study = optuna.create_study(
        direction=tuning.direction,
        study_name=tuning.study_name,
        sampler=sampler(tuning),
    )
    study.optimize(objective, n_trials=tuning.n_trials)

    best_params = study.best_params
    best_mc = merge_params(base_mc, best_params)
    best_metric = float(study.best_value)

    return TuningResult(study=study, model_config=best_mc, best_metric=best_metric)


def run_xgb_tuning(
    pipeline_config: HistoryBasedPipelineConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tuning_override: Optional[ModelTuningConfig] = None,
) -> TuningResult:
    """Convenience wrapper for ``run_model_tuning(..., 'xgb', ...)``."""
    return run_model_tuning(
        pipeline_config,
        "xgb",
        train_df,
        val_df,
        tuning_override=tuning_override,
    )


def run_ncf_tuning(
    pipeline_config: HistoryBasedPipelineConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tuning_override: Optional[ModelTuningConfig] = None,
) -> TuningResult:
    """Convenience wrapper for ``run_model_tuning(..., 'ncf', ...)``."""
    return run_model_tuning(
        pipeline_config,
        "ncf",
        train_df,
        val_df,
        tuning_override=tuning_override,
    )


def run_knowledge_tracing_tuning(
    pipeline_config: HistoryBasedPipelineConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tuning_override: Optional[ModelTuningConfig] = None,
) -> TuningResult:
    """Convenience wrapper for ``run_model_tuning(..., 'knowledge_tracing', ...)``."""
    return run_model_tuning(
        pipeline_config,
        "knowledge_tracing",
        train_df,
        val_df,
        tuning_override=tuning_override,
    )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    from pedago_ml.features.dataloader import DataLoader
    from pedago_ml.features.feature_engineering import FeatureEngineer
    from pedago_ml.features.splitter import DataSplitter
    from pedago_ml.utils.logger import logger

    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for an ML model block."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pedago_ml_configs/history_based.yaml",
        help="Pipeline YAML path",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("xgb", "ncf", "knowledge_tracing"),
        required=True,
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    app_pipeline_cfg = HistoryBasedPipelineConfig.load_config(cfg_path)
    selected_model_key = cast(ModelKey, args.model)
    selected_model_config: ModelConfig
    if selected_model_key == "xgb":
        selected_model_config = app_pipeline_cfg.xgb_config
    elif selected_model_key == "ncf":
        selected_model_config = app_pipeline_cfg.ncf_config
    else:
        selected_model_config = app_pipeline_cfg.knowledge_tracing_config
    if selected_model_config.tuning is None:
        print(
            f"No `tuning` block for {args.model} in {cfg_path}; add one to the YAML.",
            file=sys.stderr,
        )
        sys.exit(1)
    cli_tuning_override = selected_model_config.tuning.model_copy(
        update={"enabled": True}
    )

    data = DataLoader(config=app_pipeline_cfg).run()
    splitter = DataSplitter(config=app_pipeline_cfg)
    split_train_df, split_val_df, split_test_df, _encoder = splitter.run(data)
    feature_engineer = FeatureEngineer(config=app_pipeline_cfg)
    (
        train_features,
        val_features,
        test_features,
        _,
        _,
    ) = feature_engineer.run(
        train_df=split_train_df,
        val_df=split_val_df,
        test_df=split_test_df,
    )

    # Run tuning
    result = run_model_tuning(
        app_pipeline_cfg,
        selected_model_key,
        train_features,
        val_features,
        tuning_override=cli_tuning_override,
    )
    logger.info("Best metric value: {}", result.best_metric)
    logger.info(
        "Best model_params: {}",
        result.model_config.model_params.model_dump(),
    )
    best_model_config = result.model_config

    best_model: BaseMLModel
    if selected_model_key == "xgb":
        best_model = XGBModel(
            config=app_pipeline_cfg,
            model_config=cast(XGBModelConfig, best_model_config),
        )
        best_model.fit(train_features)
    elif selected_model_key == "ncf":
        best_model = NCFModel(
            config=app_pipeline_cfg,
            model_config=cast(NCFModelConfig, best_model_config),
        )
        best_model.fit(train_features, val_features)
    else:
        best_model = KnowledgeTracingModel(
            config=app_pipeline_cfg,
            model_config=cast(KnowledgeTracingModelConfig, best_model_config),
        )
        best_model.fit(train_features)

    test_ranked = best_model.rank_candidates(test_features)
    logger.info("Top ML model ranked rows (head):\n{}", test_ranked.head())

    zpd_cfg = app_pipeline_cfg.zpd_config.model_copy(deep=True)
    zpd_cfg.probability_col = best_model.model_config.output.score_col
    zpd_model = ZPDTheory(config=app_pipeline_cfg, model_config=zpd_cfg)
    zpd_ranked = zpd_model.rank_candidates(test_ranked)
    zpd_metrics = zpd_model.evaluate(zpd_ranked, top_k=2)
    logger.info("ZPD model metrics: {}", zpd_metrics)
