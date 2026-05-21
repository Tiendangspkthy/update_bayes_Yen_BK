from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedago_ml.utils.config import GridPosteriorConfig


@dataclass(frozen=True)
class PosteriorState:
    """Discrete theta grid and normalized posterior probabilities."""

    theta_grid: np.ndarray
    probs: np.ndarray


class GridPosteriorUpdater:
    """Bayesian posterior updater on a fixed theta grid."""

    def __init__(self, config: GridPosteriorConfig) -> None:
        self.config = config
        self.theta_grid = np.linspace(
            float(config.theta_min),
            float(config.theta_max),
            int(config.n_grid),
            dtype=np.float32,
        )

        if len(self.theta_grid) < 3:
            raise ValueError("n_grid must be >= 3.")
        if float(config.prior_std) <= 0:
            raise ValueError("prior_std must be > 0.")

    def initialize(self) -> PosteriorState:
        """Initialize theta prior with Normal(prior_mean, prior_std)."""
        mu = float(self.config.prior_mean)
        sigma = float(self.config.prior_std)
        z = (self.theta_grid - mu) / sigma
        probs = np.exp(-0.5 * (z**2)).astype(np.float64)
        probs = probs / probs.sum()
        return PosteriorState(
            theta_grid=self.theta_grid.copy(), probs=probs.astype(float)
        )

    def update(
        self, posterior: PosteriorState, beta_q: float, response: int
    ) -> PosteriorState:
        """Update posterior with one Bernoulli observation."""
        if response not in {0, 1}:
            raise ValueError("response must be binary (0/1).")
        if len(posterior.theta_grid) != len(posterior.probs):
            raise ValueError("theta_grid and probs lengths do not match.")

        p = 1.0 / (1.0 + np.exp(-(posterior.theta_grid - float(beta_q))))
        likelihood = p if response == 1 else (1.0 - p)
        probs = posterior.probs * likelihood
        total = probs.sum()
        if total <= 0:
            raise ValueError("Posterior underflow: probabilities sum to zero.")
        probs = probs / total
        return PosteriorState(
            theta_grid=posterior.theta_grid.copy(), probs=probs.astype(float)
        )
