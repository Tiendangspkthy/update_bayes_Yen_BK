import numpy as np
import pandas as pd

from pedago_ml.base.edu_theory import BaseEduTheory
from pedago_ml.utils.config import FlowTheoryConfig, HistoryBasedPipelineConfig
from pedago_ml.utils.logger import logger


class FlowTheory(BaseEduTheory):
    """
    Flow-based challenge balancing for candidate question ranking.

    Uses predicted correctness probability to estimate challenge level:
    - over_challenged: probability too low (too hard)
    - in_flow: probability in the target challenge band
    - under_challenged: probability too high (too easy)
    """

    def __init__(
        self, config: HistoryBasedPipelineConfig, model_config: FlowTheoryConfig
    ) -> None:
        super().__init__(config=config, model_config=model_config)
        self.params = model_config.model_params
        self._validate_config()

    def _validate_config(self) -> None:
        lower = self.params.lower_probability
        target = self.params.target_probability
        upper = self.params.upper_probability

        if not 0.0 <= lower < target < upper <= 1.0:
            raise ValueError(
                "Flow probability thresholds must satisfy "
                "0 <= lower_probability < target_probability < upper_probability <= 1."
            )

        if not 0.0 <= self.params.min_in_flow_score <= 1.0:
            raise ValueError("min_in_flow_score must be between 0 and 1.")

        if self.params.outside_tolerance <= 0:
            raise ValueError("outside_tolerance must be greater than 0.")

        total_weight = self.params.base_score_weight + self.params.flow_score_weight
        if total_weight <= 0:
            raise ValueError(
                "base_score_weight + flow_score_weight must be greater than 0."
            )

    def _prepare_probability(self, df: pd.DataFrame) -> np.ndarray:
        prob_col = self.model_config.probability_col
        if prob_col is None:
            raise ValueError("probability_col must be configured for flow theory.")

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

    def _bounded_linear_score(
        self,
        values: np.ndarray,
        lower: float,
        target: float,
        upper: float,
    ) -> np.ndarray:
        eps = 1e-8
        min_score = self.params.min_in_flow_score
        outside_tolerance = self.params.outside_tolerance

        left_width = max(target - lower, eps)
        right_width = max(upper - target, eps)

        center_width = np.where(values <= target, left_width, right_width)
        center_distance = np.abs(values - target) / center_width
        in_flow_score = 1.0 - (1.0 - min_score) * np.clip(center_distance, 0.0, 1.0)

        below_distance = np.maximum(lower - values, 0.0) / outside_tolerance
        above_distance = np.maximum(values - upper, 0.0) / outside_tolerance
        below_score = min_score * (1.0 - below_distance)
        above_score = min_score * (1.0 - above_distance)

        score = np.where(
            values < lower,
            below_score,
            np.where(values > upper, above_score, in_flow_score),
        )
        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def _make_labels(self, proba: np.ndarray) -> np.ndarray:
        labels = np.full(len(proba), "in_flow", dtype=object)
        labels[proba < self.params.lower_probability] = "over_challenged"
        labels[proba > self.params.upper_probability] = "under_challenged"
        return labels

    def _resolve_base_score(self, df: pd.DataFrame) -> np.ndarray:
        base_score_col = self.model_config.base_score_col
        if base_score_col is not None:
            self._require_columns(df, [base_score_col])
            return df[base_score_col].astype(float).to_numpy()

        prob_col = self.model_config.probability_col
        if prob_col is None:
            raise ValueError("probability_col must be configured for flow theory.")
        self._require_columns(df, [prob_col])
        return df[prob_col].astype(float).to_numpy()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input dataframe must not be empty.")

        logger.info("Applying FlowTheory scoring with probability mode")

        proba = self._prepare_probability(df)
        out = df.copy()
        out[self.output.flow_value_col] = proba
        out[self.output.flow_distance_col] = np.abs(
            proba - self.params.target_probability
        ).astype(np.float32)
        out[self.output.flow_score_col] = self._bounded_linear_score(
            values=proba,
            lower=self.params.lower_probability,
            target=self.params.target_probability,
            upper=self.params.upper_probability,
        )
        out[self.output.flow_label_col] = self._make_labels(proba)
        out[self.output.is_aligned_col] = out[self.output.flow_label_col].eq("in_flow")

        base_score = self._resolve_base_score(out)
        flow_score = out[self.output.flow_score_col].astype(float).to_numpy()
        total_weight = self.params.base_score_weight + self.params.flow_score_weight
        adjusted_score = (
            self.params.base_score_weight * base_score
            + self.params.flow_score_weight * flow_score
        ) / total_weight
        out[self.output.adjusted_score_col] = adjusted_score.astype(np.float32)
        return out

    def evaluate(
        self,
        df: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        if self.output.is_aligned_col not in df.columns:
            scored_df = self.rank_candidates(df)
        else:
            scored_df = df.copy()

        if top_k is not None:
            if self.output.rank_col not in scored_df.columns:
                scored_df = self.rank_candidates(scored_df)
            scored_df = scored_df[scored_df[self.output.rank_col] <= top_k]

        labels = scored_df[self.output.flow_label_col]
        n_rows = max(len(scored_df), 1)
        metrics = {
            "flow_alignment_rate": float(scored_df[self.output.is_aligned_col].mean()),
            "mean_flow_score": float(scored_df[self.output.flow_score_col].mean()),
            "over_challenged_rate": float(labels.eq("over_challenged").sum() / n_rows),
            "under_challenged_rate": float(
                labels.eq("under_challenged").sum() / n_rows
            ),
        }

        scope = f"top-{top_k}" if top_k is not None else "all"
        logger.info("=" * 20 + f" FLOW {scope} alignment metrics " + "=" * 20)
        for key, val in metrics.items():
            logger.info("{:<24}: {}".format(key, val))
        return metrics
