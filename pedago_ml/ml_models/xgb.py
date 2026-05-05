from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.utils.config import HistoryBasedPipelineConfig, XGBModelConfig


class XGBModel:
    """Simple XGBoost baseline for history-based question recommendation."""

    def __init__(self, config: XGBModelConfig):
        self.config = config
        self.feature_columns: list[str] = []
        self.model = XGBClassifier(**config.model_params.model_dump())

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        if self.config.feature_columns is not None:
            return self.config.feature_columns

        exclude_cols = {
            self.config.user_id,
            self.config.item_id,
            self.config.target_col,
            self.config.output.score_col,
            self.config.output.prediction_col,
            self.config.output.rank_col,
        }

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not feature_cols:
            raise ValueError("No numeric feature columns found for XGBModel.")

        return feature_cols

    def _make_x(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        return (
            df[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    def _make_y(self, df: pd.DataFrame) -> np.ndarray:
        if self.config.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")

        return df[self.config.target_col].astype(int).to_numpy()

    def fit(self, train_df: pd.DataFrame) -> "XGBModel":
        self.feature_columns = self._select_features(train_df)
        self.model.fit(self._make_x(train_df), self._make_y(train_df))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("Call fit() before predict_proba().")

        return self.model.predict_proba(self._make_x(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.config.threshold).astype(int)

    def add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.config.output.score_col] = self.predict_proba(out)
        out[self.config.output.prediction_col] = self.predict(out)
        return out

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank hidden candidate questions inside each student pool."""
        out = self.add_predictions(df)
        out[self.config.output.rank_col] = out.groupby(self.config.user_id)[
            self.config.output.score_col
        ].rank(ascending=False, method="first")

        return out.sort_values(
            [self.config.user_id, self.config.output.rank_col]
        ).reset_index(drop=True)

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        y_true = self._make_y(df)
        y_score = self.predict_proba(df)
        y_pred = (y_score >= self.config.threshold).astype(int)

        metrics = {
            "logloss": float(log_loss(y_true, y_score, labels=[0, 1])),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        metrics["auc"] = (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBModel":
        return joblib.load(path)


if __name__ == "__main__":
    pipeline_config = HistoryBasedPipelineConfig.load_config(
        "pedago_ml_configs/history_based.yaml"
    )
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

    model_config = pipeline_config.xgb_config
    model = XGBModel(config=model_config)
    model.fit(train_features)

    val_metrics = model.evaluate(val_features)
    test_ranked = model.rank_candidates(test_features)
    print("Validation Metrics:", val_metrics)
    print("Top 5 Ranked Candidates in Test Set:")
    print(test_ranked.head())
