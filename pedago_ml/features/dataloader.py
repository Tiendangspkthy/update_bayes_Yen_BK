from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from pedago_ml.utils.config import BasePipelineConfig
from pedago_ml.utils.logger import logger


class DataLoader:
    """Load and preprocess interaction and item metadata."""

    def __init__(self, config: BasePipelineConfig):
        self.config = config

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file with basic validation."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading CSV from {file_path}")
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        return df

    def transform_item_df(self, item_df: pd.DataFrame) -> pd.DataFrame:
        """Transform item metadata into model-ready item features."""
        df = item_df.copy()
        item_id_col = "ItemID"

        if df[item_id_col].duplicated().any():
            duplicated_ids = (
                df.loc[df[item_id_col].duplicated(), item_id_col].unique().tolist()
            )
            raise ValueError(
                f"Duplicated {item_id_col} found in item_df: {duplicated_ids[:10]}"
            )

        spec = (
            df["Specification"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .apply(lambda xs: [s.strip() for s in xs if s.strip()])
        )

        df["Level"] = spec.apply(lambda xs: xs[-1] if len(xs) >= 1 else None)
        df["Topic"] = spec.apply(lambda xs: xs[-2] if len(xs) >= 2 else None)
        df["CategoryList"] = spec.apply(lambda xs: xs[:-2] if len(xs) > 2 else [])

        mlb = MultiLabelBinarizer()
        cat_mat = mlb.fit_transform(df["CategoryList"])

        cat_columns = [
            f"{str(c).strip().lower().replace(' ', '_')}_category" for c in mlb.classes_
        ]

        cat_df = pd.DataFrame(cat_mat, columns=cat_columns, index=df.index)

        df = pd.concat(
            [df.drop(columns=["Specification", "CategoryList"]), cat_df],
            axis=1,
        )

        df = df.rename(columns={"ItemID": self.config.item_id})
        df[self.config.item_id] = df[self.config.item_id].astype(int)

        return df

    def get_data(self) -> pd.DataFrame:
        """Load, transform, and return interaction-level data."""
        interaction_df = self._load_csv(self.config.interaction_data_path)
        item_raw_df = self._load_csv(self.config.item_data_path)
        item_df = self.transform_item_df(item_raw_df)

        item_columns = [
            col for col in interaction_df.columns if col != self.config.user_id
        ]

        if not item_columns:
            raise ValueError("No item response columns found in interaction_df.")

        full_data = interaction_df.melt(
            id_vars=[self.config.user_id],
            value_vars=item_columns,
            var_name=self.config.item_id,
            value_name=self.config.target,
        )

        full_data[self.config.item_id] = full_data[self.config.item_id].astype(int)
        full_data = full_data.merge(
            item_df,
            on=self.config.item_id,
            how="inner",
            validate="many_to_one",
        )

        logger.info(f"Data shape: {full_data.shape}")
        logger.info(
            f"Label distribution:\n{full_data[self.config.target].value_counts()}"
        )

        return full_data

    def run(self) -> pd.DataFrame:
        """Run the data loading and preprocessing pipeline."""
        logger.info("Loading and preprocessing data...")
        return self.get_data()


if __name__ == "__main__":
    pipeline_config = BasePipelineConfig.load_config(
        "pedago_ml_configs/pipelines/history_based.yaml"
    )
    data = DataLoader(config=pipeline_config).run()
    print(data.head())
