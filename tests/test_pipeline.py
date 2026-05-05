import pytest
from conftest import BaseTest

from pedago_ml.edu_theory.zpd import ZPDTheory
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.ml_models.ncf import NCFModel
from pedago_ml.ml_models.xgb import XGBModel
from pedago_ml.utils.config import HistoryBasedPipelineConfig

config = HistoryBasedPipelineConfig.load_config("pedago_ml_configs/history_based.yaml")


class TestPipeline(BaseTest):
    def data_loader(self):
        data = DataLoader(config=config).run()
        assert data is not None
        assert len(data) > 0
        assert config.user_id in data.columns
        assert config.item_id in data.columns
        assert config.target in data.columns
        return data

    def data_splitter(self):
        data = self.data_loader()
        splitter = DataSplitter(config=config)
        train_df, val_df, test_df, encoder = splitter.run(data)
        assert train_df is not None
        assert val_df is not None
        assert test_df is not None
        assert encoder is not None
        assert len(train_df) > 0
        return train_df, val_df, test_df, encoder

    def feature_engineer(self):
        train_df, val_df, test_df, encoder = self.data_splitter()
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
        assert train_features is not None
        assert val_features is not None
        assert test_features is not None
        assert len(train_features) > 0
        assert len(val_features) > 0
        assert len(test_features) > 0
        return train_features, val_features, test_features

    def xgb_model(self):
        train_features, _, _ = self.feature_engineer()
        xgb_model_config = config.xgb_config
        xgb_model = XGBModel(config=config, model_config=xgb_model_config)
        xgb_model.fit(train_features)
        assert xgb_model is not None
        assert xgb_model.model is not None
        return xgb_model

    def ncf_model(self):
        train_features, val_features, _ = self.feature_engineer()
        ncf_model_config = config.ncf_config
        ncf_model = NCFModel(config=config, model_config=ncf_model_config)
        ncf_model.fit(train_features, val_features)
        assert ncf_model is not None
        assert ncf_model.model is not None
        return ncf_model

    def zpd_model(self, ml_model):
        _, val_features, test_features = self.feature_engineer()
        test_ranked = ml_model.rank_candidates(test_features)
        assert test_ranked is not None

        zpd_model_config = config.zpd_config
        zpd_model = ZPDTheory(config=config, model_config=zpd_model_config)
        ranked = zpd_model.rank_candidates(test_ranked)
        assert ranked is not None
        assert zpd_model.evaluate_alignment(ranked, top_k=2) is not None
        return zpd_model

    @pytest.mark.order("last")
    def test_pipeline(self):
        xgb_model = self.xgb_model()
        ncf_model = self.ncf_model()
        zpd_model = self.zpd_model(xgb_model)
        assert zpd_model is not None
