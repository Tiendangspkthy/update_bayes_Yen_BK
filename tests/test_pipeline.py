import pytest
from conftest import BaseTest

from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.utils.config import HistoryBasedPipelineConfig


class TestPipeline(BaseTest):
    @pytest.mark.order("last")
    def test_pipeline(self):
        config = HistoryBasedPipelineConfig.load_config(
            "pedago_ml_configs/history_based.yaml"
        )

        data = DataLoader(config=config).run()

        splitter = DataSplitter(config=config)
        train_df, val_df, test_df, encoder = splitter.run(data)

        feature_engineer = FeatureEngineer(config=config)
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
        assert (
            not train_features.isnull().any().any()
        ), "Null values found in train features"
        assert (
            not val_features.isnull().any().any()
        ), "Null values found in val features"
        assert (
            not test_features.isnull().any().any()
        ), "Null values found in test features"
        assert (
            set(train_features.columns)
            == set(val_features.columns)
            == set(test_features.columns)
        ), "Train/Val/Test feature columns do not match"
        assert (
            config.target in train_features.columns
        ), "Target column missing from train features"
