from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import pandas as pd

from pedago_ml.utils.config import BasePipelineConfig

ConfigT = TypeVar("ConfigT", bound=BasePipelineConfig)


@dataclass
class PipelineArtifacts:
    """
    Standard output container for history-based and cold-start pipelines.

    ranked_df:
        Primary tabular result (ranked candidates, step traces, etc.).
    metrics:
        Aggregate metrics produced by evaluate() or run().
    extras:
        Optional artifacts (fitted models, summaries, difficulty tables).
    """

    ranked_df: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class BasePipeline(ABC, Generic[ConfigT]):
    """Shared interface for end-to-end adaptive question-selection pipelines."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self.user_col = config.user_id
        self.item_col = config.item_id
        self.target_col = config.target

    @abstractmethod
    def run(self) -> PipelineArtifacts:
        """Execute the full pipeline and return artifacts."""

    @abstractmethod
    def evaluate(self, artifacts: PipelineArtifacts) -> dict[str, float]:
        """Compute evaluation metrics from pipeline artifacts."""

    def run_and_evaluate(self) -> PipelineArtifacts:
        """Run the pipeline, evaluate artifacts, and attach metrics."""
        artifacts = self.run()
        metrics = self.evaluate(artifacts)
        artifacts.metrics.update(metrics)
        return artifacts
