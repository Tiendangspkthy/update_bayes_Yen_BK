from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.base.pipeline import BasePipeline, PipelineArtifacts
from pedago_ml.base.tuning import ModelKey, get_subconfig
from pedago_ml.edu_theory.flow_theory import FlowTheory
from pedago_ml.edu_theory.zpd import ZPDTheory
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.utils.config import (
    FlowTheoryConfig,
    HistoryBasedPipelineConfig,
    ZPDTheoryConfig,
)

DEFAULT_OUTPUT_DIR = Path("outputs/history_based")


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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
    ) -> None:
        super().__init__(config=config)
        self.model_key = model_key
        self.top_k = top_k
        self.apply_zpd = apply_zpd
        self.apply_flow = apply_flow

        self.data_loader = DataLoader(config=config)
        self.splitter = DataSplitter(config=config)
        self.feature_engineer = FeatureEngineer(config=config)

    def _fit_model(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
    ) -> BaseMLModel:
        model_config, model_cls, needs_val_fit = get_subconfig(
            self.config, self.model_key
        )
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

        ml_model = self._fit_model(train_features, val_features)
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

        metrics: dict[str, float] = {}
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
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ) -> Path:
        """Write evaluation metrics and ranked test predictions to disk."""
        out_path = Path(output_dir) / self.model_key
        out_path.mkdir(parents=True, exist_ok=True)

        metrics_path = out_path / "eval_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(artifacts.metrics, metrics_file, indent=2)

        test_path = out_path / "test_ranked.csv"
        artifacts.ranked_df.to_csv(test_path, index=False)

        return out_path


if __name__ == "__main__":
    import argparse

    from pedago_ml.utils.logger import logger

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
        help="Directory for eval_metrics.json and test_ranked.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k for ZPD / Flow alignment metrics",
    )
    parser.add_argument(
        "--no-zpd",
        action="store_true",
        help="Skip ZPD reranking (ML-only or Flow-only output)",
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
    )
    pipeline_artifacts = pipeline.run()
    saved_dir = pipeline.save_outputs(
        pipeline_artifacts,
        output_dir=cli_args.output_dir,
    )

    logger.info("Saved outputs to {}", saved_dir)
    logger.info("  - eval_metrics.json")
    logger.info("  - test_ranked.csv ({} rows)", len(pipeline_artifacts.ranked_df))
    for metric_name, metric_value in sorted(pipeline_artifacts.metrics.items()):
        logger.info("{}: {}", metric_name, metric_value)
