import numpy as np
import pandas as pd

from pedago_ml.utils.config import BasePipelineConfig
from pedago_ml.utils.logger import logger


class FeatureEngineer:
    """
    Build leakage-safe aggregate features for history-based adaptive question selection.
    Feature design:
    - Train: use leave-one-out student/item aggregates to avoid target leakage.
    - Val/Test: use full-train aggregates only.
    """

    def __init__(self, config: BasePipelineConfig):
        self.config = config

        self.student_col = config.user_id
        self.item_col = config.item_id
        self.target_col = config.target

        self.smoothing = config.smoothing

    def _validate_required_columns(self, df: pd.DataFrame, split_name: str) -> None:
        required_cols = [self.student_col, self.item_col, self.target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"{split_name} is missing required columns: {missing_cols}"
            )

    def _validate_inputs(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        if train_df.empty:
            raise ValueError("train_df must not be empty.")

        for split_name, df in {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        }.items():
            self._validate_required_columns(df, split_name)

            if not df.empty and df[self.target_col].isna().any():
                raise ValueError(
                    f"{split_name}.{self.target_col} contains missing values."
                )

    def _build_full_train_aggregates(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build full-train aggregate tables for students and items."""
        student_full = (
            train_df.groupby(self.student_col)[self.target_col]
            .agg(
                student_total_correct="sum",
                student_total_attempts="count",
            )
            .reset_index()
        )

        item_full = (
            train_df.groupby(self.item_col)[self.target_col]
            .agg(
                item_total_correct="sum",
                item_total_attempts="count",
            )
            .reset_index()
        )

        return student_full, item_full

    def _add_smoothed_accuracy(
        self,
        df: pd.DataFrame,
        prefix: str,
        global_accuracy: float,
    ) -> pd.DataFrame:
        correct_col = f"{prefix}_n_correct"
        attempt_col = f"{prefix}_n_attempts"
        accuracy_col = f"{prefix}_accuracy"

        if self.smoothing == 0:
            df[accuracy_col] = np.where(
                df[attempt_col] > 0,
                df[correct_col] / df[attempt_col],
                global_accuracy,
            )
        else:
            df[accuracy_col] = (df[correct_col] + self.smoothing * global_accuracy) / (
                df[attempt_col] + self.smoothing
            )

        return df

    def _build_train_leave_one_out_features(
        self,
        train_df: pd.DataFrame,
        student_full: pd.DataFrame,
        item_full: pd.DataFrame,
        global_accuracy: float,
    ) -> pd.DataFrame:
        """
        Build train features using leave-one-out logic.

        For each train row:
        - student aggregate excludes current response
        - item aggregate excludes current response
        """
        train_out = train_df.copy()

        train_out = train_out.merge(
            student_full,
            on=self.student_col,
            how="left",
            validate="many_to_one",
        )

        train_out = train_out.merge(
            item_full,
            on=self.item_col,
            how="left",
            validate="many_to_one",
        )

        train_out["student_n_correct"] = (
            train_out["student_total_correct"] - train_out[self.target_col]
        )
        train_out["student_n_attempts"] = train_out["student_total_attempts"] - 1

        train_out["item_n_correct"] = (
            train_out["item_total_correct"] - train_out[self.target_col]
        )
        train_out["item_n_attempts"] = train_out["item_total_attempts"] - 1

        train_out = self._add_smoothed_accuracy(
            df=train_out,
            prefix="student",
            global_accuracy=global_accuracy,
        )

        train_out = self._add_smoothed_accuracy(
            df=train_out,
            prefix="item",
            global_accuracy=global_accuracy,
        )

        train_out = train_out.drop(
            columns=[
                "student_total_correct",
                "student_total_attempts",
                "student_n_attempts",
                "item_total_correct",
                "item_total_attempts",
            ]
        )

        return self._cast_feature_dtypes(train_out)

    def _build_feature_tables(
        self,
        student_full: pd.DataFrame,
        item_full: pd.DataFrame,
        global_accuracy: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build full-train feature tables for hidden val/test rows."""
        student_feature_table = student_full.rename(
            columns={
                "student_total_correct": "student_n_correct",
                "student_total_attempts": "student_n_attempts",
            }
        ).copy()

        student_feature_table = self._add_smoothed_accuracy(
            df=student_feature_table,
            prefix="student",
            global_accuracy=global_accuracy,
        )
        student_feature_table = student_feature_table.drop(
            columns=["student_n_attempts"]
        )

        item_feature_table = item_full.rename(
            columns={
                "item_total_correct": "item_n_correct",
                "item_total_attempts": "item_n_attempts",
            }
        ).copy()

        item_feature_table = self._add_smoothed_accuracy(
            df=item_feature_table,
            prefix="item",
            global_accuracy=global_accuracy,
        )

        student_feature_table = self._cast_feature_dtypes(student_feature_table)
        item_feature_table = self._cast_feature_dtypes(item_feature_table)

        return student_feature_table, item_feature_table

    def _apply_hidden_features(
        self,
        df: pd.DataFrame,
        student_feature_table: pd.DataFrame,
        item_feature_table: pd.DataFrame,
        global_accuracy: float,
    ) -> pd.DataFrame:
        """
        Apply full-train aggregate features to val/test hidden interactions.

        In this dataset, unknown students/items should not normally occur because:
        - every student has train interactions
        - every item should appear in train across many students

        Fallback is still kept for safety.
        """
        out = df.copy()

        out = out.merge(
            student_feature_table,
            on=self.student_col,
            how="left",
            validate="many_to_one",
        )

        out = out.merge(
            item_feature_table,
            on=self.item_col,
            how="left",
            validate="many_to_one",
        )

        count_cols = [
            "student_n_correct",
            "item_n_correct",
            "item_n_attempts",
        ]

        accuracy_cols = [
            "student_accuracy",
            "item_accuracy",
        ]

        for col in count_cols:
            out[col] = out[col].fillna(0)

        for col in accuracy_cols:
            out[col] = out[col].fillna(global_accuracy)

        return self._cast_feature_dtypes(out).reset_index(drop=True)

    def _cast_feature_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast engineered numeric features to float32 for model efficiency."""
        feature_cols = [
            "student_n_correct",
            "student_accuracy",
            "item_n_correct",
            "item_n_attempts",
            "item_accuracy",
        ]

        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        return df

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run leakage-safe feature engineering."""
        logger.info("Building leakage-safe aggregate features...")

        self._validate_inputs(train_df, val_df, test_df)

        global_accuracy = float(train_df[self.target_col].mean())

        if np.isnan(global_accuracy):
            raise ValueError("Global train accuracy is NaN.")

        logger.info(f"Global train accuracy: {global_accuracy:.4f}")
        logger.info(f"Feature smoothing: {self.smoothing}")

        student_full, item_full = self._build_full_train_aggregates(train_df)

        train_out = self._build_train_leave_one_out_features(
            train_df=train_df,
            student_full=student_full,
            item_full=item_full,
            global_accuracy=global_accuracy,
        )

        student_feature_table, item_feature_table = self._build_feature_tables(
            student_full=student_full,
            item_full=item_full,
            global_accuracy=global_accuracy,
        )

        val_out = self._apply_hidden_features(
            df=val_df,
            student_feature_table=student_feature_table,
            item_feature_table=item_feature_table,
            global_accuracy=global_accuracy,
        )

        test_out = self._apply_hidden_features(
            df=test_df,
            student_feature_table=student_feature_table,
            item_feature_table=item_feature_table,
            global_accuracy=global_accuracy,
        )

        logger.info(f"Train feature shape: {train_out.shape}")
        logger.info(f"Val feature shape: {val_out.shape}")
        logger.info(f"Test feature shape: {test_out.shape}")
        logger.info(f"Student feature table shape: {student_feature_table.shape}")
        logger.info(f"Item feature table shape: {item_feature_table.shape}")

        return (
            train_out.reset_index(drop=True),
            val_out.reset_index(drop=True),
            test_out.reset_index(drop=True),
            student_feature_table.reset_index(drop=True),
            item_feature_table.reset_index(drop=True),
        )
