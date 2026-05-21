import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from pedago_ml.utils.config import ColdStartPipelineConfig, HistoryBasedPipelineConfig

RANDOM_STATE = 12


class ColdStartSplitter:
    """
    Split students into calibration and evaluation sets for cold-start generalization.
    """

    def __init__(self, config: ColdStartPipelineConfig) -> None:
        self.config = config
        self.user_id = config.user_id
        self.calibration_ratio = config.cold_start_splitter.calibration_ratio
        self.seed = config.cold_start_splitter.seed

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split students into calibration and evaluation dataframes."""
        students = df[self.user_id].dropna().unique()
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(students)
        n_calib = int(len(shuffled) * self.calibration_ratio)
        calib_students = set(shuffled[:n_calib].tolist())
        calibration_df = df[df[self.user_id].isin(calib_students)].reset_index(
            drop=True
        )
        eval_df = df[~df[self.user_id].isin(calib_students)].reset_index(drop=True)
        return calibration_df, eval_df


class DataSplitter:
    """Split data into train/val/test sets with student-level grouping and encoding."""

    def __init__(self, config: HistoryBasedPipelineConfig):
        self.config = config
        self.train_ratio = config.splitter.train_ratio
        self.val_ratio = config.splitter.val_ratio
        self.test_ratio = config.splitter.test_ratio
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio

        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, but got {total:.4f}")

    def _make_student_seed(self, student_id: object) -> int:
        """Create a stable per-student random seed."""
        seed_text = f"{RANDOM_STATE}_{student_id}".encode("utf-8")
        return int(hashlib.md5(seed_text).hexdigest(), 16) % (2**32)

    def _shuffle_student_group(
        self,
        student_id: object,
        group: pd.DataFrame,
    ) -> pd.DataFrame:
        """Shuffle one student's interactions with a stable student-specific seed."""
        seed = self._make_student_seed(student_id)
        return group.sample(frac=1.0, random_state=seed).copy()

    def _get_split_indices(self, n_rows: int) -> tuple[int, int]:
        """Calculate train and validation split boundaries."""
        train_end = max(int(np.floor(n_rows * self.train_ratio)), 1)
        val_end = max(
            int(np.floor(n_rows * (self.train_ratio + self.val_ratio))),
            train_end,
        )

        return train_end, val_end

    def _split_student_group(
        self,
        student_id: object,
        group: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split one student's shuffled interactions into train/val/test."""
        shuffled_group = self._shuffle_student_group(student_id, group)

        if len(shuffled_group) == 1:
            empty_group = pd.DataFrame(columns=group.columns)
            return shuffled_group, empty_group, empty_group

        train_end, val_end = self._get_split_indices(len(shuffled_group))

        return (
            shuffled_group.iloc[:train_end],
            shuffled_group.iloc[train_end:val_end],
            shuffled_group.iloc[val_end:],
        )

    def _concat_or_empty(
        self,
        parts: list[pd.DataFrame],
        columns: pd.Index,
    ) -> pd.DataFrame:
        """Concatenate dataframe parts or return an empty dataframe."""
        if not parts:
            return pd.DataFrame(columns=columns)

        return pd.concat(parts, axis=0).reset_index(drop=True)

    def split_per_student_random(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split each student's item interactions into observed-history train set,
        validation hidden set, and test hidden set.

        This is intended for history-based next-question recommendation,
        not cold-start student-level generalization.
        """
        train_parts = []
        val_parts = []
        test_parts = []

        for student_id, group in df.groupby(self.config.user_id, sort=False):
            train_group, val_group, test_group = self._split_student_group(
                student_id=student_id,
                group=group,
            )

            train_parts.append(train_group)

            if not val_group.empty:
                val_parts.append(val_group)

            if not test_group.empty:
                test_parts.append(test_group)

        return (
            self._concat_or_empty(train_parts, df.columns),
            self._concat_or_empty(val_parts, df.columns),
            self._concat_or_empty(test_parts, df.columns),
        )

    def _get_categorical_columns(self) -> list[str]:
        """Return configured categorical item feature columns."""
        return [
            col.name for col in self.config.item_features if col.type == "categorical"
        ]

    def encode_after_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OneHotEncoder]:
        """
        Fit encoder on train only, then transform val/test.
        """
        categorical_cols = self._get_categorical_columns()

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )

        encoder.fit(train_df[categorical_cols])
        feature_names = encoder.get_feature_names_out(categorical_cols)

        def transform(df: pd.DataFrame) -> pd.DataFrame:
            encoded = encoder.transform(df[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df.index,
            )

            output = pd.concat(
                [df.drop(columns=categorical_cols), encoded_df],
                axis=1,
            )
            return output.reset_index(drop=True)

        return (
            transform(train_df),
            transform(val_df),
            transform(test_df),
            encoder,
        )

    def run(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OneHotEncoder]:
        """Run the full splitting and encoding pipeline."""
        train_df, val_df, test_df = self.split_per_student_random(df)
        return self.encode_after_split(train_df, val_df, test_df)
