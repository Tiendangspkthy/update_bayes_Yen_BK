from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from pedago_ml.utils.config import BasePipelineConfig


class BaseEduTheory(ABC):
    """Shared interface for educational theory scoring and ranking."""

    def __init__(self, config: BasePipelineConfig, model_config: Any) -> None:
        self.config = config
        self.model_config = model_config
        self.user_col = self.config.user_id
        self.item_col = self.config.item_id
        self.output = self.model_config.output

    def _require_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input dataframe is missing columns: {missing_cols}")

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add theory-specific score and label columns."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, top_k: int | None = None) -> dict[str, float]:
        """Evaluate theory alignment quality."""

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank candidate questions per learner by adjusted score."""
        out = self.apply(df)
        self._require_columns(out, [self.user_col, self.output.adjusted_score_col])
        out[self.output.rank_col] = out.groupby(self.user_col)[
            self.output.adjusted_score_col
        ].rank(method="first", ascending=False)
        return out.sort_values([self.user_col, self.output.rank_col]).reset_index(
            drop=True
        )
