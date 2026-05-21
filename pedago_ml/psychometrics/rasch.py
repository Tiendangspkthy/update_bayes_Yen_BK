from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.utils.config import ColdStartPipelineConfig, RaschModelConfig


@dataclass(frozen=True)
class _ItemStats:
    question_ids: np.ndarray
    n_correct: np.ndarray
    n_attempts: np.ndarray


class RaschModel(BaseMLModel):
    """
    Rasch (IRT-1PL) item-difficulty calibration using smoothed logit rates.

    This is a robust baseline for fixed, small item banks.
    """

    def __init__(
        self,
        config: ColdStartPipelineConfig,
        model_config: RaschModelConfig,
    ) -> None:
        super().__init__(config=config, model_config=model_config)
        self.params = model_config.model_params
        self.ability_col = model_config.ability_col
        self._difficulty_map: dict[int, float] = {}

    def _fit_item_stats(self, train_df: pd.DataFrame) -> _ItemStats:
        grouped = train_df.groupby(self.item_col)[self.target_col].agg(
            n_correct="sum", n_attempts="count"
        )
        return _ItemStats(
            question_ids=grouped.index.to_numpy(dtype=int),
            n_correct=grouped["n_correct"].to_numpy(dtype=np.float64),
            n_attempts=grouped["n_attempts"].to_numpy(dtype=np.float64),
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        *_args: Any,
        **_kwargs: Any,
    ) -> "RaschModel":
        """Estimate item difficulties from calibration interactions."""
        if train_df.empty:
            raise ValueError("train_df must not be empty.")

        stats = self._fit_item_stats(train_df)
        alpha = float(self.params.difficulty_prior_std)
        clip = float(self.params.clip_difficulty)

        p = (stats.n_correct + alpha) / (stats.n_attempts + 2.0 * alpha)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        beta = -np.log(p / (1.0 - p))
        beta = np.clip(beta, -clip, clip)

        self._difficulty_map = {
            int(qid): float(diff) for qid, diff in zip(stats.question_ids, beta)
        }
        return self

    def _difficulty_for_df(self, df: pd.DataFrame) -> np.ndarray:
        if not self._difficulty_map:
            raise RuntimeError("Call fit() before using RaschModel.")
        diffs = (
            df[self.item_col]
            .astype(int)
            .map(self._difficulty_map)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        return diffs

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        ability = df[self.ability_col].astype(float).to_numpy(dtype=np.float64)
        difficulty = self._difficulty_for_df(df)
        logits = ability - difficulty
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        y_true = df[self.target_col].astype(int).to_numpy()
        y_score = self.predict_proba(df)
        return {
            "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
            "mean_probability": float(y_score.mean()),
        }

    def get_item_difficulty(self) -> pd.DataFrame:
        """Return fitted item difficulties as a dataframe."""
        if not self._difficulty_map:
            raise RuntimeError("Call fit() before get_item_difficulty().")
        return pd.DataFrame(
            {
                self.item_col: list(self._difficulty_map.keys()),
                self.model_config.output.difficulty_col: list(
                    self._difficulty_map.values()
                ),
            }
        )
