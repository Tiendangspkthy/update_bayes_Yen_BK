import optuna
import pytest
from conftest import BaseTest

from pedago_ml.edu_theory.flow_theory import FlowTheory
from pedago_ml.edu_theory.zpd import ZPDTheory
from pedago_ml.features.dataloader import DataLoader
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.ml_models.knowledge_tracing import KnowledgeTracingModel
from pedago_ml.ml_models.ncf import NCFModel
from pedago_ml.ml_models.xgb import XGBModel
from pedago_ml.tuning.trial_executor import run_xgb_tuning
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
        train_features, val_features, _ = self.feature_engineer()
        xgb_model_config = config.xgb_config
        xgb_model = XGBModel(config=config, model_config=xgb_model_config)
        xgb_model.fit(train_features)
        assert xgb_model is not None
        assert xgb_model.model is not None
        metrics = xgb_model.evaluate(val_features)
        assert {"logloss", "accuracy", "f1", "auc"} <= set(metrics.keys())
        return xgb_model

    def ncf_model(self):
        train_features, val_features, _ = self.feature_engineer()
        ncf_model_config = config.ncf_config
        ncf_model = NCFModel(config=config, model_config=ncf_model_config)
        ncf_model.fit(train_features, val_features)
        assert ncf_model is not None
        assert ncf_model.model is not None
        metrics = ncf_model.evaluate(val_features)
        assert {"log_loss", "accuracy", "auc"} <= set(metrics.keys())
        return ncf_model

    def knowledge_tracing_model(self):
        train_features, val_features, _ = self.feature_engineer()
        kt_model_config = config.knowledge_tracing_config
        kt_model = KnowledgeTracingModel(config=config, model_config=kt_model_config)
        kt_model.fit(train_features)
        assert kt_model is not None
        metrics = kt_model.evaluate(val_features)
        assert {"log_loss", "accuracy", "auc"} <= set(metrics.keys())
        return kt_model

    def zpd_model(self, ml_model):
        _, val_features, test_features = self.feature_engineer()
        test_ranked = ml_model.rank_candidates(test_features)
        assert test_ranked is not None

        # Use the current ML model's probability output column (xgb/ncf/kt)
        zpd_model_config = config.zpd_config.model_copy(deep=True)
        zpd_model_config.probability_col = ml_model.model_config.output.score_col
        zpd_model = ZPDTheory(config=config, model_config=zpd_model_config)
        ranked = zpd_model.rank_candidates(test_ranked)
        assert ranked is not None
        alignment = zpd_model.evaluate(ranked, top_k=2)
        assert alignment is not None
        return zpd_model

    def flow_model(self, ml_model):
        _, _, test_features = self.feature_engineer()
        test_ranked = ml_model.rank_candidates(test_features)
        assert test_ranked is not None

        # Use the current ML model's probability output column (xgb/ncf/kt)
        flow_model_config = config.flow_config.model_copy(deep=True)
        flow_model_config.probability_col = ml_model.model_config.output.score_col
        flow_model = FlowTheory(config=config, model_config=flow_model_config)
        ranked = flow_model.rank_candidates(test_ranked)
        assert ranked is not None

        alignment = flow_model.evaluate(ranked, top_k=2)
        assert {
            "flow_alignment_rate",
            "mean_flow_score",
            "over_challenged_rate",
            "under_challenged_rate",
        } <= set(alignment.keys())
        return flow_model

    def test_tuning(self):
        train_features, val_features, _ = self.feature_engineer()
        xgb_model = self.xgb_model()
        ncf_model = self.ncf_model()
        kt_model = self.knowledge_tracing_model()
        zpd_model = self.zpd_model(xgb_model)
        flow_model = self.flow_model(xgb_model)
        assert zpd_model is not None
        assert flow_model is not None
        assert ncf_model is not None
        assert kt_model is not None

        result = run_xgb_tuning(
            config,
            train_features,
            val_features,
            tuning_override=config.xgb_config.tuning,
        )
        assert isinstance(result.study, optuna.Study)
        assert result.best_metric is not None
        assert result.model_config.model_params is not None

    @pytest.mark.order("last")
    def test_pipeline(self):
        xgb_model = self.xgb_model()
        zpd_model = self.zpd_model(xgb_model)
        assert zpd_model is not None
        flow_model = self.flow_model(xgb_model)
        assert flow_model is not None

        ncf_model = self.ncf_model()
        zpd_model = self.zpd_model(ncf_model)
        assert zpd_model is not None
        flow_model = self.flow_model(ncf_model)
        assert flow_model is not None

        kt_model = self.knowledge_tracing_model()
        zpd_model = self.zpd_model(kt_model)
        assert zpd_model is not None
        flow_model = self.flow_model(kt_model)
        assert flow_model is not None
