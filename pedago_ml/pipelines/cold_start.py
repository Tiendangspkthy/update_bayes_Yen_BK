from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pedago_ml.base.pipeline import BasePipeline, PipelineArtifacts
from pedago_ml.evaluation import ColdStartEvaluator
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.splitter import ColdStartSplitter
from pedago_ml.psychometrics.ability_estimator import GridAbilityEstimator
from pedago_ml.psychometrics.posterior_updater import GridPosteriorUpdater
from pedago_ml.psychometrics.rasch import RaschModel
from pedago_ml.scoring import ColdStartSelector
from pedago_ml.utils.config import ColdStartPipelineConfig
from pedago_ml.utils.logger import (
    add_pipeline_file_log,
    logger,
    remove_pipeline_file_log,
)


class ColdStartPipeline(BasePipeline[ColdStartPipelineConfig]):
    """Cold-start diagnostic pipeline for new students."""

    def __init__(self, config: ColdStartPipelineConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.output_dir = self.config.output_dir
        self.data_loader = DataLoader(config=config)
        self.splitter = ColdStartSplitter(config=config)
        self.selector = ColdStartSelector(
            config.selector_config,
            item_col=config.item_id,
            difficulty_col=config.rasch_config.output.difficulty_col,
        )
        self.updater = GridPosteriorUpdater(config.posterior_config)
        self.estimator = GridAbilityEstimator(config.posterior_config)
        self.evaluator = ColdStartEvaluator()

    def output_path(self, output_dir: Path | str | None = None) -> Path:
        base = self.output_dir if output_dir is None else Path(output_dir)
        return base

    def save_outputs(
        self,
        artifacts: PipelineArtifacts,
        output_dir: Path | str | None = None,
    ) -> Path:
        out_path = self.output_path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        run_info: dict[str, Any] = {
            "pipeline": "cold_start",
            "user_id": self.config.user_id,
            "item_id": self.config.item_id,
            "target": self.config.target,
            "cold_start_splitter": self.config.cold_start_splitter.model_dump(),
            "posterior_config": self.config.posterior_config.model_dump(),
            "stopping_config": self.config.stopping_config.model_dump(),
            "selector_config": self.config.selector_config.model_dump(),
            "rasch_config": {
                "model_name": self.config.rasch_config.model_name,
                "model_params": self.config.rasch_config.model_params.model_dump(),
                "output": self.config.rasch_config.output.model_dump(),
            },
            "n_eval_students": int(
                artifacts.extras["student_summary_df"]["StudentID"].nunique()
            ),
            "n_step_rows": len(artifacts.extras["step_trace_df"]),
        }

        with (out_path / self.config.run_info_filename).open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(run_info, f, indent=2)

        with (out_path / "eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(artifacts.metrics, f, indent=2)

        artifacts.extras["student_summary_df"].to_csv(
            out_path / "student_summary.csv", index=False
        )
        artifacts.extras["step_trace_df"].to_csv(
            out_path / "step_trace.csv", index=False
        )
        artifacts.extras["item_difficulty_df"].to_csv(
            out_path / "item_difficulty.csv", index=False
        )

        return out_path

    def _build_item_bank(
        self, eval_df: pd.DataFrame, rasch_model: RaschModel
    ) -> pd.DataFrame:
        item_meta_cols = [
            self.config.item_id,
            "Topic",
            *[c for c in eval_df.columns if c.endswith("_category")],
        ]
        item_meta = (
            eval_df[item_meta_cols].drop_duplicates(subset=[self.config.item_id]).copy()
        )
        item_meta = item_meta.merge(
            rasch_model.get_item_difficulty(),
            on=self.config.item_id,
            how="left",
            validate="one_to_one",
        )
        old_difficulty_col = rasch_model.model_config.output.difficulty_col
        new_difficulty_col = self.config.rasch_config.output.difficulty_col
        item_meta = item_meta.rename(columns={old_difficulty_col: new_difficulty_col})
        return item_meta

    def _estimate_oracle_theta(
        self,
        student_df: pd.DataFrame,
        item_bank: pd.DataFrame,
    ) -> float:
        posterior = self.updater.initialize()
        lookup = item_bank.set_index(self.config.item_id)[
            self.config.rasch_config.output.difficulty_col
        ].to_dict()
        for _, row in student_df.iterrows():
            beta = float(lookup[int(row[self.config.item_id])])
            posterior = self.updater.update(
                posterior, beta_q=beta, response=int(row[self.config.target])
            )
        summary = self.estimator.summarize(posterior)
        if self.config.posterior_config.estimator == "map":
            return summary.theta_map
        return summary.theta_mean

    def _run_single_student_episode(
        self,
        student_id: int,
        student_truth_df: pd.DataFrame,
        item_bank_df: pd.DataFrame,
    ) -> dict[str, Any]:
        posterior = self.updater.initialize()
        asked = pd.DataFrame(columns=item_bank_df.columns)
        step_rows: list[dict[str, Any]] = []
        lookup = student_truth_df.set_index(self.config.item_id)[
            self.config.target
        ].to_dict()

        step_idx = 0
        while True:
            step_idx += 1
            asked_ids = set(asked[self.config.item_id].tolist())
            candidates = item_bank_df[
                ~item_bank_df[self.config.item_id].isin(asked_ids)
            ].copy()
            if candidates.empty:
                break

            choice = self.selector.select_next_question(candidates, posterior, asked)
            top = choice.scored_candidates.iloc[0]
            qid = int(top[self.config.item_id])
            response = int(lookup[qid])
            beta_q = float(top[self.config.rasch_config.output.difficulty_col])

            prior_summary = self.estimator.summarize(posterior)
            posterior = self.updater.update(posterior, beta_q=beta_q, response=response)
            post_summary = self.estimator.summarize(posterior)

            asked = pd.concat(
                [asked, candidates[candidates[self.config.item_id] == qid]],
                axis=0,
                ignore_index=True,
            )
            step_rows.append(
                {
                    "StudentID": student_id,
                    "step_idx": step_idx,
                    "QuestionID": qid,
                    "Topic": top.get("Topic", None),
                    "response": response,
                    "expected_p": float(top["expected_p"]),
                    "expected_info": float(top["expected_info"]),
                    "zpd_distance": float(top["zpd_distance"]),
                    "in_zpd": bool(top["in_zpd"]),
                    "similarity_penalty": float(top["similarity_penalty"]),
                    "final_score": float(top["final_score"]),
                    "posterior_variance_before": float(prior_summary.variance),
                    "posterior_variance_after": float(post_summary.variance),
                }
            )

            should_stop = (
                step_idx >= self.config.stopping_config.min_questions
                and post_summary.variance
                <= self.config.stopping_config.variance_threshold
            ) or step_idx >= self.config.stopping_config.max_questions
            if should_stop:
                break

        final_summary = self.estimator.summarize(posterior)
        return {
            "summary": final_summary,
            "step_rows": step_rows,
        }

    def run(self) -> PipelineArtifacts:
        full_df = self.data_loader.run()
        calibration_df, eval_df = self.splitter.run(full_df)

        rasch = RaschModel(self.config, self.config.rasch_config)
        rasch.fit(calibration_df)
        item_bank = self._build_item_bank(eval_df, rasch)

        summary_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []

        for student_id, student_df in eval_df.groupby(self.config.user_id):
            episode = self._run_single_student_episode(
                int(student_id),
                student_df[[self.config.item_id, self.config.target]].copy(),
                item_bank,
            )
            estimate = episode["summary"]
            theta_est = (
                estimate.theta_map
                if self.config.posterior_config.estimator == "map"
                else estimate.theta_mean
            )
            theta_oracle = self._estimate_oracle_theta(student_df, item_bank)
            summary_rows.append(
                {
                    "StudentID": int(student_id),
                    "theta_estimate": float(theta_est),
                    "theta_oracle": float(theta_oracle),
                    "variance": float(estimate.variance),
                    "ci_low": float(estimate.ci_low),
                    "ci_high": float(estimate.ci_high),
                    "n_questions": len(episode["step_rows"]),
                }
            )
            step_rows.extend(episode["step_rows"])

        student_summary_df = pd.DataFrame(summary_rows)
        step_trace_df = pd.DataFrame(step_rows)
        metrics = self.evaluate(
            PipelineArtifacts(
                ranked_df=step_trace_df,
                metrics={},
                extras={
                    "student_summary_df": student_summary_df,
                    "step_trace_df": step_trace_df,
                    "item_difficulty_df": rasch.get_item_difficulty(),
                },
            )
        )

        return PipelineArtifacts(
            ranked_df=step_trace_df,
            metrics=metrics,
            extras={
                "student_summary_df": student_summary_df,
                "step_trace_df": step_trace_df,
                "item_difficulty_df": rasch.get_item_difficulty(),
            },
        )

    def evaluate(self, artifacts: PipelineArtifacts) -> dict[str, float]:
        summary_df = artifacts.extras["student_summary_df"]
        trace_df = artifacts.extras["step_trace_df"]
        metrics = {}
        metrics.update(self.evaluator.evaluate_student_summary(summary_df))
        metrics.update(self.evaluator.evaluate_step_trace(trace_df))
        return metrics


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = "pedago_ml_configs/cold_start.yaml"
    pipeline_config = ColdStartPipelineConfig.load_config(DEFAULT_CONFIG_PATH)
    pipeline = ColdStartPipeline(pipeline_config)

    out_path = pipeline_config.output_dir
    log_filename = out_path / pipeline_config.log_filename
    log_sink_id = add_pipeline_file_log(log_filename)

    try:
        artifacts = pipeline.run()
        saved_dir = pipeline.save_outputs(
            artifacts, output_dir=pipeline_config.output_dir
        )

        summary_df = artifacts.extras["student_summary_df"]
        trace_df = artifacts.extras["step_trace_df"]

        logger.info("Saved outputs to {}", out_path)
        logger.info("  - eval_metrics.json")
        logger.info("  - {}", out_path / pipeline_config.run_info_filename)
        logger.info("  - student_summary.csv ({} students)", len(summary_df))
        logger.info("  - step_trace.csv ({} rows)", len(trace_df))
        logger.info("  - item_difficulty.csv")
        logger.info("  - {}", log_filename)
        for metric_name, metric_value in sorted(artifacts.metrics.items()):
            logger.info("{}: {}", metric_name, metric_value)
    finally:
        remove_pipeline_file_log(log_sink_id)
