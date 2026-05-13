from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pedago_ml.base.ml_model import BaseMLModel
from pedago_ml.utils.config import (
    HistoryBasedPipelineConfig,
    KnowledgeTracingModelConfig,
)
from pedago_ml.utils.logger import logger


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class _MasteryTable:
    """Vectorized lookup table for user-category mastery."""

    user_to_idx: dict[Any, int]
    category_cols: list[str]
    mastery: np.ndarray  # shape: (n_users, n_categories)
    global_mastery: float

    def get_user_indices(self, user_ids: pd.Series) -> np.ndarray:
        return user_ids.map(self.user_to_idx).fillna(-1).astype(np.int64).to_numpy()


class KnowledgeTracingModel(BaseMLModel):
    """
    Lightweight knowledge tracing baseline tracking per-category mastery.

    It uses Beta-Binomial smoothing for mastery and a logistic link to output
    predicted correctness probability:
      pred_proba = sigmoid(w0 + w1 * mastery_category + w2 * item_accuracy)
    """

    def __init__(
        self,
        config: HistoryBasedPipelineConfig,
        model_config: KnowledgeTracingModelConfig,
    ) -> None:
        super().__init__(config=config, model_config=model_config)
        self.params = model_config.model_params
        self._mastery_table: _MasteryTable | None = None

    def _resolve_category_cols(self, df: pd.DataFrame) -> list[str]:
        wanted = self.params.category_columns
        lower_map = {c.lower(): c for c in df.columns}

        resolved: list[str] = []
        for col in wanted:
            actual = lower_map.get(col.lower())
            if actual is not None:
                resolved.append(actual)

        if not resolved:
            raise ValueError(
                "No category columns found in dataframe. "
                f"Configured: {wanted}, available example: {list(df.columns)[:10]}"
            )
        return resolved

    def fit(self, train_df: pd.DataFrame) -> "KnowledgeTracingModel":
        if train_df.empty:
            raise ValueError("train_df must not be empty.")
        if self.target_col not in train_df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        if self.user_col not in train_df.columns:
            raise ValueError(f"Missing user column: {self.user_col}")

        category_cols = self._resolve_category_cols(train_df)
        alpha = float(self.params.alpha)
        beta = float(self.params.beta)
        denom_prior = alpha + beta
        if denom_prior <= 0:
            raise ValueError("alpha + beta must be > 0.")

        users = train_df[self.user_col].dropna().unique().tolist()
        user_to_idx = {u: i for i, u in enumerate(sorted(users))}
        n_users = len(user_to_idx)
        n_cat = len(category_cols)

        # Compute global mastery as fallback
        y = train_df[self.target_col].astype(float).to_numpy()
        global_mastery = float((alpha + y.sum()) / (alpha + beta + len(y)))

        # For each category, compute user-level attempts and correct counts
        mastery = np.full((n_users, n_cat), global_mastery, dtype=np.float32)

        train_tmp = train_df[[self.user_col, self.target_col] + category_cols].copy()
        train_tmp[self.target_col] = train_tmp[self.target_col].astype(float)

        for j, cat_col in enumerate(category_cols):
            mask = train_tmp[cat_col].fillna(0).astype(int) > 0
            if not mask.any():
                continue

            cat_df = train_tmp.loc[mask, [self.user_col, self.target_col]]
            grouped = cat_df.groupby(self.user_col)[self.target_col].agg(
                n_correct="sum", n_attempts="count"
            )

            for user_id, row in grouped.iterrows():
                idx = user_to_idx.get(user_id)
                if idx is None:
                    continue
                n_correct = float(row["n_correct"])
                n_attempts = float(row["n_attempts"])
                mastery[idx, j] = (alpha + n_correct) / (alpha + beta + n_attempts)

        self._mastery_table = _MasteryTable(
            user_to_idx=user_to_idx,
            category_cols=category_cols,
            mastery=mastery.astype(np.float32),
            global_mastery=global_mastery,
        )

        logger.info(
            "Fitted KnowledgeTracingModel with "
            f"{n_users} users and {n_cat} categories. "
            f"global_mastery={global_mastery:.4f}"
        )
        return self

    def _mastery_for_rows(self, df: pd.DataFrame) -> np.ndarray:
        table = self._mastery_table
        if table is None:
            raise RuntimeError("Call fit() before predict_proba().")

        self._require_columns(df, table.category_cols + [self.user_col])
        user_idx = table.get_user_indices(df[self.user_col])
        cat_mat = df[table.category_cols].fillna(0).astype(np.float32).to_numpy()

        # unknown users -> use global mastery
        mastery_lookup = np.where(
            user_idx[:, None] >= 0,
            table.mastery[np.clip(user_idx, 0, table.mastery.shape[0] - 1)],
            table.global_mastery,
        ).astype(np.float32)

        active = (cat_mat > 0).astype(np.float32)
        active_sum = active.sum(axis=1, keepdims=True)

        if self.params.category_aggregation == "max":
            masked = np.where(active > 0, mastery_lookup, -np.inf)
            m = np.max(masked, axis=1)
            m[~np.isfinite(m)] = table.global_mastery
            return m.astype(np.float32)

        # mean over active categories; fallback to global mastery if none active
        active_sum = np.where(active_sum == 0, 1.0, active_sum)
        m = (mastery_lookup * active).sum(axis=1) / active_sum.squeeze(1)
        no_active = active.sum(axis=1) == 0
        m[no_active] = table.global_mastery
        return m.astype(np.float32)

    def _require_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing columns: {missing}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        table = self._mastery_table
        if table is None:
            raise RuntimeError("Call fit() before predict_proba().")

        mastery = self._mastery_for_rows(df)

        item_acc_col = self.params.item_accuracy_col
        if item_acc_col not in df.columns:
            raise ValueError(f"Missing item accuracy column: {item_acc_col}")
        item_acc = df[item_acc_col].astype(float).to_numpy(dtype=np.float32)

        logits = (
            float(self.params.w0)
            + float(self.params.w1) * mastery
            + float(self.params.w2) * item_acc
        ).astype(np.float32)

        proba = _sigmoid(logits).astype(np.float32)
        if self.params.clip_probability:
            proba = np.clip(proba, 0.0, 1.0).astype(np.float32)
        return proba

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")
        y_true = df[self.target_col].astype(int).to_numpy()
        y_score = self.predict_proba(df)
        y_pred = (y_score >= self.threshold).astype(int)

        metrics = {
            "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        metrics["auc"] = (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )

        logger.info("=" * 20 + " KT METRICS " + "=" * 20)
        for key, val in metrics.items():
            logger.info("{:<20}: {}".format(key, val))
        return metrics
