from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedago_ml.psychometrics.posterior_updater import PosteriorState
from pedago_ml.utils.config import GridPosteriorConfig


@dataclass(frozen=True)
class AbilityEstimate:
    """Posterior summary statistics for student ability."""

    theta_mean: float
    theta_map: float
    variance: float
    ci_low: float
    ci_high: float

    @property
    def theta(self) -> float:
        """Default ability point estimate (posterior mean)."""
        return self.theta_mean


class GridAbilityEstimator:
    """Summarize posterior distribution into scalar diagnostics."""

    def __init__(self, config: GridPosteriorConfig) -> None:
        self.config = config

    def summarize(self, posterior: PosteriorState) -> AbilityEstimate:
        """Compute mean, MAP, variance, and normal-approx CI on the theta grid."""
        grid = posterior.theta_grid.astype(float)
        probs = posterior.probs.astype(float)
        probs = probs / probs.sum()

        theta_mean = float(np.sum(grid * probs))
        theta_map = float(grid[int(np.argmax(probs))])
        variance = float(np.sum(((grid - theta_mean) ** 2) * probs))
        std = float(np.sqrt(max(variance, 0.0)))
        ci_low = theta_mean - 1.96 * std
        ci_high = theta_mean + 1.96 * std
        return AbilityEstimate(
            theta_mean=theta_mean,
            theta_map=theta_map,
            variance=variance,
            ci_low=ci_low,
            ci_high=ci_high,
        )
