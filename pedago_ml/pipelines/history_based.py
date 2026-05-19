from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.base.pipeline import BasePipeline, PipelineArtifacts
from pedago_ml.base.tuning import (
    ModelConfig,
    ModelKey,
    SavedTuningCheckpoint,
    TuningResult,
    get_subconfig,
    load_tuning_checkpoint,
    save_tuning_checkpoint,
)
from pedago_ml.edu_theory.flow_theory import FlowTheory
from pedago_ml.edu_theory.zpd import ZPDTheory
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.tuning import run_model_tuning
from pedago_ml.utils.config import (
    FlowTheoryConfig,
    HistoryBasedPipelineConfig,
    ZPDTheoryConfig,
)
from pedago_ml.utils.logger import (
    add_pipeline_file_log,
    logger,
    remove_pipeline_file_log,
)

DEFAULT_OUTPUT_DIR = Path("outputs/history_based")
LOG_FILENAME = "pipeline.log"
TUNING_FILENAME = "tuning_result.json"
RUN_INFO_FILENAME = "run_info.json"

ParamSource = Literal["auto", "config", "tuned", "tune"]
ResolvedParamSource = Literal["config", "tuned", "tune"]


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _output_path(output_dir: Path | str, model_key: str) -> Path:
    return Path(output_dir) / model_key


@dataclass(frozen=True)
class ResolvedTrainingParams:
    """How ML hyperparameters were chosen for this pipeline run."""

    param_source: ResolvedParamSource
    model_config: ModelConfig
    tuning_result: TuningResult | None = None
    tuning_checkpoint: SavedTuningCheckpoint | None = None


class HistoryBasedPipeline(BasePipeline[HistoryBasedPipelineConfig]):
    """End-to-end history-based adaptive question recommendation pipeline."""

    def __init__(
        self,
        config: HistoryBasedPipelineConfig,
        *,
        model_key: ModelKey = "xgb",
        top_k: int = 2,
        apply_zpd: bool = True,
        apply_flow: bool = True,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        param_source: ParamSource = "auto",
    ) -> None:
        super().__init__(config=config)
        self.model_key = model_key
        self.top_k = top_k
        self.apply_zpd = apply_zpd
        self.apply_flow = apply_flow
        self.output_dir = Path(output_dir)
        self.param_source = param_source

        self.data_loader = DataLoader(config=config)
        self.splitter = DataSplitter(config=config)
        self.feature_engineer = FeatureEngineer(config=config)

    def output_path(self, output_dir: Path | str | None = None) -> Path:
        """Directory for this model's artifacts under output_dir."""
        base = self.output_dir if output_dir is None else Path(output_dir)
        return _output_path(base, self.model_key)

    def tuning_checkpoint_path(self, output_dir: Path | str | None = None) -> Path:
        return self.output_path(output_dir) / TUNING_FILENAME

    def _yaml_tuning_enabled(self) -> bool:
        base_model_config, _, _ = get_subconfig(self.config, self.model_key)
        tuning = base_model_config.tuning
        return tuning is not None and tuning.enabled

    def resolve_param_source(
        self, output_dir: Path | str | None = None
    ) -> ResolvedParamSource:
        """Resolve effective param source (auto follows YAML + checkpoint presence)."""
        if self.param_source != "auto":
            return cast(ResolvedParamSource, self.param_source)

        if self._yaml_tuning_enabled():
            return "tune"

        if self.tuning_checkpoint_path(output_dir).is_file():
            return "tuned"

        return "config"

    def _resolve_training_params(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
    ) -> ResolvedTrainingParams:
        from pedago_ml.utils.logger import logger

        resolved = self.resolve_param_source()
        base_model_config, _, _ = get_subconfig(self.config, self.model_key)
        checkpoint_path = self.tuning_checkpoint_path()

        if resolved == "tune":
            tuning_block = base_model_config.tuning
            if tuning_block is None or not tuning_block.search_space:
                raise ValueError(
                    f"params-source=tune requires a non-empty tuning block for "
                    f"{self.model_key} in the pipeline config."
                )
            tuning_override = tuning_block.model_copy(update={"enabled": True})
            logger.info(
                "Training params: TUNE (Optuna, n_trials={}, metric={})",
                tuning_override.n_trials,
                tuning_override.metric,
            )
            tuning_result = run_model_tuning(
                self.config,
                self.model_key,
                train_features,
                val_features,
                tuning_override=tuning_override,
            )
            logger.info(
                "Tuning finished | best {}={}",
                tuning_override.metric,
                tuning_result.best_metric,
            )
            logger.info(
                "Best model_params: {}",
                tuning_result.model_config.model_params.model_dump(),
            )
            return ResolvedTrainingParams(
                param_source="tune",
                model_config=tuning_result.model_config,
                tuning_result=tuning_result,
            )

        if resolved == "tuned":
            checkpoint = load_tuning_checkpoint(
                checkpoint_path, self.config, self.model_key
            )
            logger.info(
                "Training params: TUNED (loaded best hyperparameters from {})",
                checkpoint.source_path,
            )
            logger.info(
                "Checkpoint best metric: {} | params: {}",
                checkpoint.best_metric,
                checkpoint.model_config.model_params.model_dump(),
            )
            return ResolvedTrainingParams(
                param_source="tuned",
                model_config=checkpoint.model_config,
                tuning_checkpoint=checkpoint,
            )

        logger.info(
            "Training params: CONFIG (YAML model_params from {})",
            base_model_config.model_name,
        )
        logger.info(
            "model_params: {}",
            base_model_config.model_params.model_dump(),
        )
        return ResolvedTrainingParams(
            param_source="config",
            model_config=base_model_config,
        )

    def _fit_model(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        model_config: ModelConfig,
    ) -> BaseMLModel:
        _, model_cls, needs_val_fit = get_subconfig(self.config, self.model_key)
        model = model_cls(config=self.config, model_config=model_config)
        if needs_val_fit:
            model.fit(train_features, val_features)
        else:
            model.fit(train_features)
        return model

    def _theory_configs(
        self, ml_model: BaseMLModel
    ) -> tuple[ZPDTheoryConfig, FlowTheoryConfig]:
        probability_col = ml_model.model_config.output.score_col
        zpd_cfg = self.config.zpd_config.model_copy(deep=True)
        zpd_cfg.probability_col = probability_col
        flow_cfg = self.config.flow_config.model_copy(deep=True)
        flow_cfg.probability_col = probability_col
        return zpd_cfg, flow_cfg

    def _merge_theory_columns(
        self,
        base: pd.DataFrame,
        theory_df: pd.DataFrame,
        theory_cols: list[str],
        merge_on: list[str],
    ) -> pd.DataFrame:
        """Left-join theory-specific columns without duplicating ML feature columns."""
        cols_to_add = [
            col
            for col in theory_cols
            if col in theory_df.columns and col not in base.columns
        ]
        if not cols_to_add:
            return base
        return base.merge(
            theory_df[merge_on + cols_to_add],
            on=merge_on,
            how="left",
            validate="one_to_one",
        )

    def build_test_export_df(self, extras: dict[str, Any]) -> pd.DataFrame:
        """
        Merge ML, ZPD, and Flow scores into one export table.

        ``ranked_df`` used for evaluation may be ZPD-only; this export keeps all layers.
        """
        ml_df = extras["ml_test_ranked"].copy()
        ml_model: BaseMLModel = extras["ml_model"]
        ml_out = ml_model.model_config.output
        merge_on = [self.user_col, self.item_col]

        if ml_out.rank_col in ml_df.columns:
            ml_df = ml_df.rename(columns={ml_out.rank_col: "ml_rank"})

        export_df = ml_df
        zpd_ranked = extras.get("zpd_ranked")
        flow_ranked = extras.get("flow_ranked")

        if zpd_ranked is not None:
            zpd_out = self.config.zpd_config.output
            export_df = self._merge_theory_columns(
                export_df,
                zpd_ranked,
                [
                    zpd_out.zpd_value_col,
                    zpd_out.zpd_distance_col,
                    zpd_out.zpd_score_col,
                    zpd_out.zpd_label_col,
                    zpd_out.is_aligned_col,
                    zpd_out.adjusted_score_col,
                    zpd_out.rank_col,
                ],
                merge_on,
            )

        if flow_ranked is not None:
            flow_out = self.config.flow_config.output
            export_df = self._merge_theory_columns(
                export_df,
                flow_ranked,
                [
                    flow_out.flow_value_col,
                    flow_out.flow_distance_col,
                    flow_out.flow_score_col,
                    flow_out.flow_label_col,
                    flow_out.is_aligned_col,
                    flow_out.adjusted_score_col,
                    flow_out.rank_col,
                ],
                merge_on,
            )

        if zpd_ranked is not None:
            final_rank_col = self.config.zpd_config.output.rank_col
        elif flow_ranked is not None:
            final_rank_col = self.config.flow_config.output.rank_col
        else:
            final_rank_col = "ml_rank"

        export_df["final_rank"] = export_df[final_rank_col]
        return export_df.sort_values(
            [self.user_col, "final_rank"], kind="mergesort"
        ).reset_index(drop=True)

    def run(self) -> PipelineArtifacts:
        full_df = self.data_loader.run()
        train_df, val_df, test_df, encoder = self.splitter.run(full_df)
        (
            train_features,
            val_features,
            test_features,
            _,
            _,
        ) = self.feature_engineer.run(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        training = self._resolve_training_params(train_features, val_features)
        ml_model = self._fit_model(
            train_features,
            val_features,
            model_config=training.model_config,
        )
        ml_val_ranked = ml_model.rank_candidates(val_features)
        ml_test_ranked = ml_model.rank_candidates(test_features)

        zpd_cfg, flow_cfg = self._theory_configs(ml_model)
        zpd_model = ZPDTheory(config=self.config, model_config=zpd_cfg)
        flow_model = FlowTheory(config=self.config, model_config=flow_cfg)

        zpd_ranked: pd.DataFrame | None = None
        flow_ranked: pd.DataFrame | None = None
        ranked_df = ml_test_ranked

        if self.apply_zpd:
            zpd_ranked = zpd_model.rank_candidates(ml_test_ranked)
            ranked_df = zpd_ranked
        if self.apply_flow:
            flow_ranked = flow_model.rank_candidates(ml_test_ranked)
            if not self.apply_zpd:
                ranked_df = flow_ranked

        extras: dict[str, Any] = {
            "model_key": self.model_key,
            "param_source": training.param_source,
            "training": training,
            "ml_model": ml_model,
            "zpd_model": zpd_model if self.apply_zpd else None,
            "flow_model": flow_model if self.apply_flow else None,
            "encoder": encoder,
            "train_features": train_features,
            "val_features": val_features,
            "test_features": test_features,
            "ml_val_ranked": ml_val_ranked,
            "ml_test_ranked": ml_test_ranked,
            "zpd_ranked": zpd_ranked,
            "flow_ranked": flow_ranked,
            "test_export_df": self.build_test_export_df(
                {
                    "ml_test_ranked": ml_test_ranked,
                    "ml_model": ml_model,
                    "zpd_ranked": zpd_ranked,
                    "flow_ranked": flow_ranked,
                }
            ),
        }

        metrics = self.evaluate(
            PipelineArtifacts(
                ranked_df=ranked_df,
                metrics={},
                extras=extras,
            )
        )

        return PipelineArtifacts(
            ranked_df=ranked_df,
            metrics=metrics,
            extras=extras,
        )

    def evaluate(self, artifacts: PipelineArtifacts) -> dict[str, float]:
        extras = artifacts.extras
        ml_model = extras["ml_model"]
        val_features = extras["val_features"]
        test_features = extras["test_features"]
        training: ResolvedTrainingParams = extras["training"]

        metrics: dict[str, float] = {}
        if training.tuning_result is not None:
            metrics["tuning_best_metric"] = float(training.tuning_result.best_metric)
        elif training.tuning_checkpoint is not None:
            metrics["tuning_best_metric"] = float(
                training.tuning_checkpoint.best_metric
            )

        metrics.update(_prefix_metrics(ml_model.evaluate(val_features), "val_ml"))
        metrics.update(_prefix_metrics(ml_model.evaluate(test_features), "test_ml"))

        zpd_ranked = extras.get("zpd_ranked")
        zpd_model = extras.get("zpd_model")
        if zpd_ranked is not None and zpd_model is not None:
            metrics.update(
                _prefix_metrics(
                    zpd_model.evaluate(zpd_ranked, top_k=self.top_k),
                    f"zpd_top{self.top_k}",
                )
            )

        flow_ranked = extras.get("flow_ranked")
        flow_model = extras.get("flow_model")
        if flow_ranked is not None and flow_model is not None:
            metrics.update(
                _prefix_metrics(
                    flow_model.evaluate(flow_ranked, top_k=self.top_k),
                    f"flow_top{self.top_k}",
                )
            )

        return metrics

    def save_outputs(
        self,
        artifacts: PipelineArtifacts,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Write eval artifacts, run metadata, and optional tuning checkpoint."""
        out_path = self.output_path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        training: ResolvedTrainingParams = artifacts.extras["training"]
        run_info: dict[str, Any] = {
            "model_key": self.model_key,
            "param_source": training.param_source,
            "param_source_requested": self.param_source,
            "yaml_tuning_enabled": self._yaml_tuning_enabled(),
            "model_params": training.model_config.model_params.model_dump(),
        }
        if training.tuning_checkpoint is not None:
            run_info["tuning_checkpoint_path"] = str(
                training.tuning_checkpoint.source_path
            )
            run_info["tuning_best_metric"] = training.tuning_checkpoint.best_metric

        test_export_df = artifacts.extras.get("test_export_df")
        if test_export_df is None:
            test_export_df = self.build_test_export_df(artifacts.extras)
        run_info["test_ranked_columns"] = {
            "ml_score_col": artifacts.extras["ml_model"].model_config.output.score_col,
            "includes_zpd": artifacts.extras.get("zpd_ranked") is not None,
            "includes_flow": artifacts.extras.get("flow_ranked") is not None,
            "final_rank_source": (
                self.config.zpd_config.output.rank_col
                if artifacts.extras.get("zpd_ranked") is not None
                else (
                    self.config.flow_config.output.rank_col
                    if artifacts.extras.get("flow_ranked") is not None
                    else "ml_rank"
                )
            ),
        }

        metrics_payload = {
            "param_source": training.param_source,
            **artifacts.metrics,
        }
        with (out_path / "eval_metrics.json").open(
            "w", encoding="utf-8"
        ) as metrics_file:
            json.dump(metrics_payload, metrics_file, indent=2)

        with (out_path / RUN_INFO_FILENAME).open(
            "w", encoding="utf-8"
        ) as run_info_file:
            json.dump(run_info, run_info_file, indent=2)

        test_export_df.to_csv(out_path / "test_ranked.csv", index=False)

        if training.tuning_result is not None:
            save_tuning_checkpoint(
                out_path / TUNING_FILENAME,
                self.model_key,
                training.tuning_result.model_config,
                training.tuning_result.best_metric,
                study=training.tuning_result.study,
            )

        return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full history-based adaptive question-selection pipeline."
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
        default="xgb",
        help="ML model to train and rank with",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for pipeline outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k for ZPD / Flow alignment metrics",
    )
    parser.add_argument(
        "--params-source",
        type=str,
        choices=("auto", "config", "tuned", "tune"),
        default="auto",
        help=(
            "How to choose ML hyperparameters: "
            "auto (YAML tuning.enabled → tune, else checkpoint → tuned, else config), "
            "tune (Optuna), tuned (load checkpoint), config (YAML model_params only)"
        ),
    )
    parser.add_argument(
        "--no-zpd",
        action="store_true",
        help="Skip ZPD reranking",
    )
    parser.add_argument(
        "--no-flow",
        action="store_true",
        help="Skip Flow reranking",
    )
    cli_args = parser.parse_args()

    pipeline_config = HistoryBasedPipelineConfig.load_config(cli_args.config)
    pipeline = HistoryBasedPipeline(
        pipeline_config,
        model_key=cast(ModelKey, cli_args.model),
        top_k=cli_args.top_k,
        apply_zpd=not cli_args.no_zpd,
        apply_flow=not cli_args.no_flow,
        output_dir=cli_args.output_dir,
        param_source=cast(ParamSource, cli_args.params_source),
    )

    out_path = pipeline.output_path()
    log_sink_id = add_pipeline_file_log(out_path / LOG_FILENAME)

    resolved_source = pipeline.resolve_param_source()
    logger.info("History-based pipeline | model={}", cli_args.model)
    logger.info("Config: {}", cli_args.config)
    logger.info("Output dir: {}", out_path)
    logger.info(
        "Params source: requested={}, resolved={}",
        cli_args.params_source,
        resolved_source,
    )
    logger.info(
        "Options: top_k={}, apply_zpd={}, apply_flow={}",
        cli_args.top_k,
        not cli_args.no_zpd,
        not cli_args.no_flow,
    )

    try:
        pipeline_artifacts = pipeline.run()
        saved_dir = pipeline.save_outputs(pipeline_artifacts)

        logger.info("Saved outputs to {}", saved_dir)
        logger.info("  - eval_metrics.json (param_source={})", resolved_source)
        logger.info("  - {}", RUN_INFO_FILENAME)
        logger.info("  - test_ranked.csv ({} rows)", len(pipeline_artifacts.ranked_df))
        logger.info("  - {}", LOG_FILENAME)
        training: ResolvedTrainingParams = pipeline_artifacts.extras["training"]
        if training.param_source == "tune":
            logger.info("  - {} (updated)", TUNING_FILENAME)
        for metric_name, metric_value in sorted(pipeline_artifacts.metrics.items()):
            logger.info("{}: {}", metric_name, metric_value)
    finally:
        remove_pipeline_file_log(log_sink_id)
