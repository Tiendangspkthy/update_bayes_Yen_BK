from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pedago_ml.features.dataloader import DataLoader as DataLoaderFeature
from pedago_ml.features.feature_engineering import FeatureEngineer
from pedago_ml.features.splitter import DataSplitter
from pedago_ml.utils.config import HistoryBasedPipelineConfig, NCFModelConfig
from pedago_ml.utils.logger import logger


class InteractionDataset(Dataset):
    """Torch dataset for student-question interactions."""

    def __init__(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        features: np.ndarray,
        target: np.ndarray | None = None,
    ) -> None:
        self.user_idx = torch.as_tensor(user_idx, dtype=torch.long)
        self.item_idx = torch.as_tensor(item_idx, dtype=torch.long)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = None

        if target is not None:
            self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.target is None:
            return self.user_idx[idx], self.item_idx[idx], self.features[idx]

        return (
            self.user_idx[idx],
            self.item_idx[idx],
            self.features[idx],
            self.target[idx],
        )


class NCFNetwork(nn.Module):
    """
    Simple Neural Collaborative Filtering model.

    Architecture:
    - student embedding
    - question embedding
    - optional numeric side features
    - MLP classifier returning logits
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_features: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2 + n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec, features], dim=1)
        return self.mlp(x).squeeze(1)


class NCFModel:
    """Trainable NCF wrapper for history-based adaptive question selection."""

    def __init__(
        self, config: HistoryBasedPipelineConfig, model_config: NCFModelConfig
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.user_col = self.config.user_id
        self.item_col = self.config.item_id
        self.target_col = self.config.target

        self.user_to_idx: dict[Any, int] = {}
        self.item_to_idx: dict[Any, int] = {}
        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

        self.model: NCFNetwork | None = None
        self.history: list[dict[str, float]] = []
        self.device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        if self.model_config.model_params.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.model_config.model_params.device

        return torch.device(device)

    def _set_seed(self) -> None:
        seed = self.model_config.model_params.random_state
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _validate_required_columns(self, df: pd.DataFrame, split_name: str) -> None:
        required_cols = [self.user_col, self.item_col]

        if self.target_col in df.columns:
            required_cols.append(self.target_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{split_name} is missing required columns: {missing_cols}"
            )

    def _resolve_feature_columns(self, train_df: pd.DataFrame) -> list[str]:
        if self.model_config.feature_columns is not None:
            missing_cols = [
                col
                for col in self.model_config.feature_columns
                if col not in train_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Configured feature_columns not found: {missing_cols}"
                )
            return list(self.model_config.feature_columns)

        return []

    def _fit_id_mappings(self, train_df: pd.DataFrame) -> None:
        users = train_df[self.user_col].dropna().unique().tolist()
        items = train_df[self.item_col].dropna().unique().tolist()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(users))}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(items))}

    def _unknown_user_idx(self) -> int:
        return len(self.user_to_idx)

    def _unknown_item_idx(self) -> int:
        return len(self.item_to_idx)

    def _encode_ids(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        user_idx = (
            df[self.user_col]
            .map(self.user_to_idx)
            .fillna(self._unknown_user_idx())
            .astype(np.int64)
            .to_numpy()
        )
        item_idx = (
            df[self.item_col]
            .map(self.item_to_idx)
            .fillna(self._unknown_item_idx())
            .astype(np.int64)
            .to_numpy()
        )
        return user_idx, item_idx

    def _fit_feature_scaler(self, train_df: pd.DataFrame) -> None:
        if not self.feature_columns:
            self.feature_mean = np.zeros(0, dtype=np.float32)
            self.feature_std = np.ones(0, dtype=np.float32)
            return

        features = (
            train_df[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        feature_mean = features.mean(axis=0).astype(np.float32)
        feature_std = features.std(axis=0).astype(np.float32)
        feature_std[feature_std == 0] = 1.0
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            return np.zeros((len(df), 0), dtype=np.float32)

        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Feature scaler is not fitted.")

        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing feature columns: {missing_cols}")

        features = (
            df[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        return ((features - self.feature_mean) / self.feature_std).astype(np.float32)

    def _build_dataset(self, df: pd.DataFrame, with_target: bool) -> InteractionDataset:
        user_idx, item_idx = self._encode_ids(df)
        features = self._encode_features(df)
        target = None

        if with_target:
            target = df[self.target_col].astype(np.float32).to_numpy()

        return InteractionDataset(
            user_idx=user_idx,
            item_idx=item_idx,
            features=features,
            target=target,
        )

    def _build_loader(
        self,
        dataset: InteractionDataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.model_config.model_params.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _init_network(self) -> None:
        params = self.model_config.model_params
        self.model = NCFNetwork(
            n_users=len(self.user_to_idx) + 1,
            n_items=len(self.item_to_idx) + 1,
            n_features=len(self.feature_columns),
            embedding_dim=params.embedding_dim,
            hidden_dims=params.hidden_dims,
            dropout=params.dropout,
        ).to(self.device)

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.train()
        total_loss = 0.0
        total_rows = 0

        for user_idx, item_idx, features, target in loader:
            user_idx = user_idx.to(self.device)
            item_idx = item_idx.to(self.device)
            features = features.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            logits = self.model(user_idx, item_idx, features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            batch_size = len(target)
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size

        return total_loss / max(total_rows, 1)

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "NCFModel":
        """Fit the NCF model."""
        self._validate_required_columns(train_df, "train_df")
        if train_df.empty:
            raise ValueError("train_df must not be empty.")

        if train_df[self.target_col].isna().any():
            raise ValueError(f"train_df.{self.target_col} contains missing values.")

        if val_df is not None and not val_df.empty:
            self._validate_required_columns(val_df, "val_df")

        self._set_seed()
        self.feature_columns = self._resolve_feature_columns(train_df)
        self._fit_id_mappings(train_df)
        self._fit_feature_scaler(train_df)
        self._init_network()
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        params = self.model_config.model_params
        train_dataset = self._build_dataset(train_df, with_target=True)
        train_loader = self._build_loader(train_dataset, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = np.inf
        best_state: dict[str, torch.Tensor] | None = None
        patience_left = params.early_stopping_patience

        logger.info(
            "Training NCF model with "
            f"{len(self.user_to_idx)} users, {len(self.item_to_idx)} items, "
            f"{len(self.feature_columns)} side features."
        )

        for epoch in range(1, params.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, criterion)
            record = {"epoch": float(epoch), "train_loss": float(train_loss)}

            if val_df is not None and not val_df.empty:
                val_metrics = self.evaluate(val_df)
                val_loss = val_metrics["log_loss"]
                record.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    patience_left = params.early_stopping_patience
                else:
                    patience_left -= 1

                if patience_left <= 0:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    self.history.append(record)
                    break

            self.history.append(record)

            if epoch % params.log_every_n_epochs == 0 or epoch == 1:
                logger.info("=" * 20 + f" EPOCH {epoch}/{params.epochs} " + "=" * 20)
                for key, val in record.items():
                    logger.info("{:<20}: {}".format(key, val))

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of correct response."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        self._validate_required_columns(df, "df")
        dataset = self._build_dataset(df, with_target=False)
        loader = self._build_loader(dataset, shuffle=False)

        self.model.eval()
        preds: list[np.ndarray] = []

        with torch.no_grad():
            for user_idx, item_idx, features in loader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                features = features.to(self.device)
                logits = self.model(user_idx, item_idx, features)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds.append(probs)

        return np.concatenate(preds).astype(np.float32)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary correctness label."""
        proba = self.predict_proba(df)
        return (proba >= self.model_config.threshold).astype(np.int64)

    def add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with probability, binary prediction, and per-user rank."""
        out = df.copy()
        out[self.model_config.output.score_col] = self.predict_proba(out)
        out[self.model_config.output.prediction_col] = (
            out[self.model_config.output.score_col] >= self.model_config.threshold
        ).astype(int)
        out[self.model_config.output.rank_col] = out.groupby(self.user_col)[
            self.model_config.output.score_col
        ].rank(method="first", ascending=False)
        return out

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Evaluate prediction quality on a labelled dataframe."""
        if self.target_col not in df.columns:
            raise ValueError(f"df must contain target column: {self.target_col}")

        y_true = df[self.target_col].astype(int).to_numpy()
        y_proba = self.predict_proba(df)
        y_pred = (y_proba >= self.model_config.threshold).astype(int)

        metrics = {
            "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

        if len(np.unique(y_true)) == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            metrics["auc"] = float("nan")

        return metrics

    def save(self, file_path: str | Path) -> None:
        """Save model weights, mappings, scaler, and training history."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Feature scaler is not fitted.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.model_config.to_dict(),
            "state_dict": self.model.state_dict(),
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
            "feature_columns": self.feature_columns,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "history": self.history,
        }
        torch.save(payload, path)
        logger.info(f"Saved NCF model to {path}")

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        pipeline_config: HistoryBasedPipelineConfig,
        model_config: NCFModelConfig | None = None,
    ) -> "NCFModel":
        """Load a saved NCF model."""
        path = Path(file_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        resolved_model_config = model_config or pipeline_config.ncf_config
        instance = cls(config=pipeline_config, model_config=resolved_model_config)
        instance.user_to_idx = payload["user_to_idx"]
        instance.item_to_idx = payload["item_to_idx"]
        instance.feature_columns = payload["feature_columns"]
        instance.feature_mean = payload["feature_mean"]
        instance.feature_std = payload["feature_std"]
        instance.history = payload.get("history", [])
        instance._init_network()
        if instance.model is None:
            raise RuntimeError("Model is not initialized.")
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        return instance


if __name__ == "__main__":
    pipeline_config = HistoryBasedPipelineConfig.load_config(
        "pedago_ml_configs/history_based.yaml"
    )
    data = DataLoaderFeature(config=pipeline_config).run()
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

    model_config = pipeline_config.ncf_config
    model = NCFModel(config=pipeline_config, model_config=model_config)
    model.fit(train_features, val_features)
    model.save("outputs/ncf_model.pth")
    model = NCFModel.load(
        "outputs/ncf_model.pth",
        pipeline_config=pipeline_config,
        model_config=model_config,
    )
    model.predict(test_features)
