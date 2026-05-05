from typing import Literal

import numpy as np
import pandas as pd

from pedago_ml.base.edu_theory import BaseEduTheory
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.ml_models.xgb import XGBModel
from pedago_ml.utils.config import HistoryBasedPipelineConfig, ZPDTheoryConfig
from pedago_ml.utils.logger import logger

ZPDMode = Literal["probability", "ability_difficulty"]


class ZPDTheory(BaseEduTheory):
    """
    Apply Zone of Proximal Development scoring to candidate questions.

    The class supports two practical scoring modes:
    - probability: use predicted probability of correctness from XGB / NCF / KT.
    - ability_difficulty: use student ability and item difficulty from Rasch / IRT.

    For history-based recommendation, probability mode is the simplest default:
    questions are considered inside ZPD when predicted correctness probability is
    neither too low nor too high.
    """

    def __init__(
        self, config: HistoryBasedPipelineConfig, model_config: ZPDTheoryConfig
    ) -> None:
        super().__init__(config=config, model_config=model_config)
        self.params = model_config.model_params
        self._validate_config()

    def _validate_config(self) -> None:
        mode = self.params.score_mode
        if mode not in {"probability", "ability_difficulty"}:
            raise ValueError(
                "model_params.score_mode must be either "
                "'probability' or 'ability_difficulty'."
            )

        if not 0.0 <= self.params.min_in_zone_score <= 1.0:
            raise ValueError("min_in_zone_score must be between 0 and 1.")

        if self.params.outside_tolerance <= 0:
            raise ValueError("outside_tolerance must be greater than 0.")

        if mode == "probability":
            lower = self.params.lower_probability
            target = self.params.target_probability
            upper = self.params.upper_probability

            if not 0.0 <= lower < target < upper <= 1.0:
                raise ValueError(
                    "Probability thresholds must satisfy "
                    "0 <= lower_probability < target_probability "
                    "< upper_probability <= 1."
                )

        if mode == "ability_difficulty":
            lower = self.params.lower_difficulty_gap
            target = self.params.target_difficulty_gap
            upper = self.params.upper_difficulty_gap

            if not lower < target < upper:
                raise ValueError(
                    "Difficulty gap thresholds must satisfy "
                    "lower_difficulty_gap < target_difficulty_gap "
                    "< upper_difficulty_gap."
                )

    def _prepare_probability(self, df: pd.DataFrame) -> np.ndarray:
        prob_col = self.model_config.probability_col
        if prob_col is None:
            raise ValueError("probability_col must be configured for probability mode.")

        self._require_columns(df, [prob_col])
        proba = df[prob_col].astype(float).to_numpy()

        if self.params.clip_probability:
            return np.clip(proba, 0.0, 1.0).astype(np.float32)

        invalid_mask = (proba < 0.0) | (proba > 1.0) | np.isnan(proba)
        if invalid_mask.any():
            raise ValueError(
                f"{self.model_config.probability_col} must be in [0, 1] and not NaN."
            )

        return proba.astype(np.float32)

    def _prepare_difficulty_gap(self, df: pd.DataFrame) -> np.ndarray:
        ability_col = self.model_config.ability_col
        difficulty_col = self.model_config.item_difficulty_col
        if ability_col is None or difficulty_col is None:
            raise ValueError(
                "ability_col and item_difficulty_col must be configured when "
                "score_mode='ability_difficulty'."
            )

        self._require_columns(df, [ability_col, difficulty_col])
        ability = df[ability_col].astype(float).to_numpy()
        difficulty = df[difficulty_col].astype(float).to_numpy()
        gap = difficulty - ability

        if np.isnan(gap).any():
            raise ValueError("Difficulty gap contains NaN values.")

        return gap.astype(np.float32)

    def _bounded_linear_score(
        self,
        values: np.ndarray,
        lower: float,
        target: float,
        upper: float,
    ) -> np.ndarray:
        """
        Convert probability/gap values into a [0, 1] ZPD score.

        Inside the target zone, score stays high and peaks at the target value.
        Outside the target zone, score decays linearly to zero.
        """
        eps = 1e-8
        min_score = self.params.min_in_zone_score
        outside_tolerance = self.params.outside_tolerance

        left_width = max(target - lower, eps)
        right_width = max(upper - target, eps)

        center_width = np.where(values <= target, left_width, right_width)
        center_distance = np.abs(values - target) / center_width
        in_zone_score = 1.0 - (1.0 - min_score) * np.clip(center_distance, 0.0, 1.0)

        below_distance = np.maximum(lower - values, 0.0) / outside_tolerance
        above_distance = np.maximum(values - upper, 0.0) / outside_tolerance
        below_score = min_score * (1.0 - below_distance)
        above_score = min_score * (1.0 - above_distance)

        score = np.where(
            values < lower,
            below_score,
            np.where(values > upper, above_score, in_zone_score),
        )

        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def _make_labels(
        self,
        values: np.ndarray,
        lower: float,
        upper: float,
        low_label: str,
        high_label: str,
    ) -> np.ndarray:
        labels = np.full(len(values), "in_zpd", dtype=object)
        labels[values < lower] = low_label
        labels[values > upper] = high_label
        return labels

    def _score_probability_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self._prepare_probability(df)
        lower = self.params.lower_probability
        target = self.params.target_probability
        upper = self.params.upper_probability

        out = df.copy()
        out[self.output.zpd_value_col] = proba
        out[self.output.zpd_distance_col] = np.abs(proba - target).astype(np.float32)
        out[self.output.zpd_score_col] = self._bounded_linear_score(
            values=proba,
            lower=lower,
            target=target,
            upper=upper,
        )
        out[self.output.zpd_label_col] = self._make_labels(
            values=proba,
            lower=lower,
            upper=upper,
            low_label="too_hard",
            high_label="too_easy",
        )
        out[self.output.is_aligned_col] = out[self.output.zpd_label_col].eq("in_zpd")
        return out

    def _score_ability_difficulty_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        gap = self._prepare_difficulty_gap(df)
        lower = self.params.lower_difficulty_gap
        target = self.params.target_difficulty_gap
        upper = self.params.upper_difficulty_gap

        out = df.copy()
        out[self.output.zpd_value_col] = gap
        out[self.output.zpd_distance_col] = np.abs(gap - target).astype(np.float32)
        out[self.output.zpd_score_col] = self._bounded_linear_score(
            values=gap,
            lower=lower,
            target=target,
            upper=upper,
        )
        out[self.output.zpd_label_col] = self._make_labels(
            values=gap,
            lower=lower,
            upper=upper,
            low_label="too_easy",
            high_label="too_hard",
        )
        out[self.output.is_aligned_col] = out[self.output.zpd_label_col].eq("in_zpd")
        return out

    def _resolve_base_score(self, df: pd.DataFrame) -> np.ndarray:
        base_score_col = self.model_config.base_score_col
        if base_score_col is not None:
            self._require_columns(df, [base_score_col])
            return df[base_score_col].astype(float).to_numpy()

        if self.params.score_mode == "probability":
            prob_col = self.model_config.probability_col
            if prob_col is None:
                raise ValueError(
                    "probability_col must be configured for probability mode."
                )
            self._require_columns(df, [prob_col])
            return df[prob_col].astype(float).to_numpy()

        return df[self.output.zpd_score_col].astype(float).to_numpy()

    def _add_adjusted_score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        base_score = self._resolve_base_score(out)
        zpd_score = out[self.output.zpd_score_col].astype(float).to_numpy()

        base_weight = self.params.base_score_weight
        zpd_weight = self.params.zpd_score_weight
        total_weight = base_weight + zpd_weight

        if total_weight <= 0:
            raise ValueError(
                "base_score_weight + zpd_score_weight must be greater than 0."
            )

        adjusted_score = (
            base_weight * base_score + zpd_weight * zpd_score
        ) / total_weight
        out[self.output.adjusted_score_col] = adjusted_score.astype(np.float32)
        return out

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ZPD value, label, score, alignment flag, and adjusted score."""
        if df.empty:
            raise ValueError("Input dataframe must not be empty.")

        logger.info(f"Applying ZPD scoring with mode={self.params.score_mode}")

        if self.params.score_mode == "probability":
            out = self._score_probability_mode(df)
        else:
            out = self._score_ability_difficulty_mode(df)

        return self._add_adjusted_score(out)

    def evaluate(
        self,
        df: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate how many selected candidates are inside the configured ZPD zone.

        If top_k is provided, evaluation is performed on top-k ranked candidates
        per student. If rank column is missing, candidates are ranked first.
        """
        if self.output.is_aligned_col not in df.columns:
            scored_df = self.rank_candidates(df)
        else:
            scored_df = df.copy()

        if top_k is not None:
            if self.output.rank_col not in scored_df.columns:
                scored_df = self.rank_candidates(scored_df)
            scored_df = scored_df[scored_df[self.output.rank_col] <= top_k]

        labels = scored_df[self.output.zpd_label_col]
        n_rows = max(len(scored_df), 1)

        metrics = {
            "zpd_alignment_rate": float(scored_df[self.output.is_aligned_col].mean()),
            "mean_zpd_score": float(scored_df[self.output.zpd_score_col].mean()),
            "too_hard_rate": float(labels.eq("too_hard").sum() / n_rows),
            "too_easy_rate": float(labels.eq("too_easy").sum() / n_rows),
        }
        scope = f"top-{top_k}" if top_k is not None else "all"
        logger.info("=" * 20 + f" ZPD {scope} alignment metrics " + "=" * 20)
        for key, val in metrics.items():
            logger.info("{:<20}: {}".format(key, val))
        return metrics

    def evaluate_alignment(
        self,
        df: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        """Backward-compatible alias for evaluate()."""
        return self.evaluate(df=df, top_k=top_k)


if __name__ == "__main__":
    pipeline_config = HistoryBasedPipelineConfig.load_config(
        "pedago_ml_configs/history_based.yaml"
    )
    model_config = pipeline_config.zpd_config

    data = DataLoader(config=pipeline_config).run()
    splitter = DataSplitter(config=pipeline_config)
    train_df, val_df, test_df, encoder = splitter.run(data)
    feature_engineer = FeatureEngineer(config=pipeline_config)
    (
        train_features,
        val_features,
        test_features,
        _,
        _,
    ) = feature_engineer.run(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    xgb_model_config = pipeline_config.xgb_config
    xgb_model = XGBModel(config=pipeline_config, model_config=xgb_model_config)
    xgb_model.fit(train_features)

    val_metrics = xgb_model.evaluate(val_features)
    test_ranked = xgb_model.rank_candidates(test_features)
    print(test_ranked.columns)

    zpd_model_config = pipeline_config.zpd_config
    zpd = ZPDTheory(config=pipeline_config, model_config=zpd_model_config)
    ranked = zpd.rank_candidates(test_ranked)
    print(ranked)
    print(zpd.evaluate_alignment(ranked, top_k=2))
