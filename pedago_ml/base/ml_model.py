from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from pedago_ml.utils.config import BasePipelineConfig


class BaseMLModel(ABC):
    """Shared interface and default behaviors for predictive ML models."""

    def __init__(self, config: BasePipelineConfig, model_config: Any) -> None:
        self.config = config
        self.model_config = model_config
        self.user_col = self.config.user_id
        self.item_col = self.config.item_id
        self.target_col = self.config.target

    @property
    def threshold(self) -> float:
        return float(self.model_config.threshold)

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, *args: Any, **kwargs: Any) -> "BaseMLModel":
        """Train model on labelled data."""

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability scores."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Evaluate model quality on labelled data."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Threshold probabilities into binary predictions."""
        return (self.predict_proba(df) >= self.threshold).astype(np.int64)

    def add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach score and prediction columns to a dataframe copy."""
        out = df.copy()
        out[self.model_config.output.score_col] = self.predict_proba(out)
        out[self.model_config.output.prediction_col] = self.predict(out)
        return out

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank candidate items by predicted score per user."""
        out = self.add_predictions(df)
        out[self.model_config.output.rank_col] = out.groupby(self.user_col)[
            self.model_config.output.score_col
        ].rank(ascending=False, method="first")
        return out.sort_values(
            [self.user_col, self.model_config.output.rank_col]
        ).reset_index(drop=True)
