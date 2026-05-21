from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pedago_ml.psychometrics.posterior_updater import PosteriorState
from pedago_ml.utils.config import ColdStartSelectorConfig


@dataclass(frozen=True)
class SelectionResult:
    """Selected question and full ranked candidate table."""

    selected_question_id: int
    scored_candidates: pd.DataFrame


class ColdStartSelector:
    """Score/select next question for cold-start diagnosis (constrained ZPD only)."""

    def __init__(
        self,
        config: ColdStartSelectorConfig,
        item_col: str = "QuestionID",
        difficulty_col: str = "item_difficulty",
    ) -> None:
        self.config = config
        self.item_col = item_col
        self.difficulty_col = difficulty_col

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _zpd_bounds(self) -> tuple[float, float, float]:
        zpd = self.config.zpd
        return (
            float(zpd.lower_probability),
            float(zpd.target_probability),
            float(zpd.upper_probability),
        )

    def _in_zpd_mask(self, expected_p: np.ndarray) -> np.ndarray:
        lower, _, upper = self._zpd_bounds()
        return (expected_p >= lower) & (expected_p <= upper)

    def _zpd_distance_to_target(self, expected_p: np.ndarray) -> np.ndarray:
        """Diagnostic distance to target probability (lower is better)."""
        _, target, _ = self._zpd_bounds()
        return np.abs(expected_p - target).astype(np.float32)

    @staticmethod
    def _category_columns(df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if col.endswith("_category")]

    def _topic_similarity_penalty(
        self,
        candidate_df: pd.DataFrame,
        recent: pd.DataFrame,
    ) -> np.ndarray:
        if "Topic" not in candidate_df.columns or "Topic" not in recent.columns:
            return np.zeros(len(candidate_df), dtype=np.float32)
        recent_topics = set(recent["Topic"].astype(str).tolist())
        return candidate_df["Topic"].astype(str).isin(recent_topics).astype(np.float32)

    def _category_similarity_penalty(
        self,
        candidate_df: pd.DataFrame,
        recent: pd.DataFrame,
    ) -> np.ndarray:
        category_cols = self._category_columns(candidate_df)
        if not category_cols or not all(col in recent.columns for col in category_cols):
            return np.zeros(len(candidate_df), dtype=np.float32)

        cand_mat = candidate_df[category_cols].astype(float).to_numpy()
        recent_mat = recent[category_cols].astype(float).to_numpy()
        recent_norm = np.linalg.norm(recent_mat, axis=1) + 1e-8
        cand_norm = np.linalg.norm(cand_mat, axis=1) + 1e-8

        max_sim = np.zeros(len(candidate_df), dtype=np.float32)
        for i in range(len(candidate_df)):
            sims = (recent_mat @ cand_mat[i]) / (recent_norm * cand_norm[i])
            max_sim[i] = float(np.max(sims))
        return max_sim

    def _similarity_penalty(
        self,
        candidate_df: pd.DataFrame,
        asked_history: pd.DataFrame,
    ) -> np.ndarray:
        """Penalty for repeating recent topics or category profiles."""
        if asked_history.empty:
            return np.zeros(len(candidate_df), dtype=np.float32)

        recent_window = int(max(self.config.similarity.recent_window, 0))
        if recent_window == 0:
            return np.zeros(len(candidate_df), dtype=np.float32)

        recent = asked_history.tail(recent_window)
        topic_penalty = self._topic_similarity_penalty(candidate_df, recent)
        category_penalty = self._category_similarity_penalty(candidate_df, recent)
        return (
            float(self.config.similarity.topic_weight) * topic_penalty
            + float(self.config.similarity.category_weight) * category_penalty
        ).astype(np.float32)

    def _expected_response_stats(
        self,
        beta: np.ndarray,
        posterior: PosteriorState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return expected P(correct) and Fisher information per candidate."""
        theta = posterior.theta_grid.astype(np.float64)
        weights = posterior.probs.astype(np.float64)
        p_theta = self._sigmoid(theta[:, None] - beta[None, :])
        expected_p = (weights[:, None] * p_theta).sum(axis=0)
        expected_info = (weights[:, None] * (p_theta * (1.0 - p_theta))).sum(axis=0)
        return expected_p, expected_info

    def score_candidates(
        self,
        candidate_df: pd.DataFrame,
        posterior: PosteriorState,
        asked_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rank candidates with constrained ZPD, information, and similarity."""
        if candidate_df.empty:
            raise ValueError("candidate_df must not be empty.")
        if self.difficulty_col not in candidate_df.columns:
            raise ValueError(
                f"candidate_df must include '{self.difficulty_col}' column."
            )

        beta = candidate_df[self.difficulty_col].astype(np.float64).to_numpy()
        expected_p, expected_info = self._expected_response_stats(beta, posterior)
        in_zpd = self._in_zpd_mask(expected_p)
        zpd_distance = self._zpd_distance_to_target(expected_p)
        similarity_penalty = self._similarity_penalty(candidate_df, asked_history)

        info_term = float(self.config.info_weight) * expected_info
        penalty_term = float(self.config.similarity.lambda_penalty) * similarity_penalty

        eligible = in_zpd
        if not bool(eligible.any()):
            eligible = np.ones(len(candidate_df), dtype=bool)

        final_score = np.full(len(candidate_df), -np.inf, dtype=np.float64)
        final_score[eligible] = info_term[eligible] - penalty_term[eligible]

        out = candidate_df.copy()
        out["expected_p"] = expected_p.astype(np.float32)
        out["expected_info"] = expected_info.astype(np.float32)
        out["zpd_distance"] = zpd_distance
        out["in_zpd"] = in_zpd.astype(bool)
        out["similarity_penalty"] = similarity_penalty.astype(np.float32)
        out["final_score"] = final_score.astype(np.float32)
        out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
        return out

    def select_next_question(
        self,
        candidate_df: pd.DataFrame,
        posterior: PosteriorState,
        asked_history: pd.DataFrame,
    ) -> SelectionResult:
        """Pick the highest-scoring candidate question."""
        scored = self.score_candidates(candidate_df, posterior, asked_history)
        qid = int(scored.iloc[0][self.item_col])
        return SelectionResult(selected_question_id=qid, scored_candidates=scored)
