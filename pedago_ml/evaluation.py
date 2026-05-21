from __future__ import annotations

import numpy as np
import pandas as pd


class ColdStartEvaluator:
    """Aggregate evaluation metrics for cold-start simulation traces."""

    def evaluate_student_summary(self, summary_df: pd.DataFrame) -> dict[str, float]:
        """Ability-estimation metrics vs oracle theta on eval students."""
        if summary_df.empty:
            raise ValueError("summary_df must not be empty.")

        delta = summary_df["theta_estimate"] - summary_df["theta_oracle"]
        rmse = float(np.sqrt(np.mean(delta**2)))
        mae = float(np.mean(np.abs(delta)))
        bias = float(np.mean(delta))
        coverage = float(
            (
                (summary_df["theta_oracle"] >= summary_df["ci_low"])
                & (summary_df["theta_oracle"] <= summary_df["ci_high"])
            ).mean()
        )
        return {
            "theta_rmse": rmse,
            "theta_mae": mae,
            "theta_bias": bias,
            "coverage_95": coverage,
            "avg_final_variance": float(summary_df["variance"].mean()),
            "avg_questions_to_stop": float(summary_df["n_questions"].mean()),
        }

    def evaluate_step_trace(self, trace_df: pd.DataFrame) -> dict[str, float]:
        """Step-level ZPD alignment, topic coverage, and similarity stats."""
        if trace_df.empty:
            return {
                "zpd_alignment_rate": 0.0,
                "topic_coverage_mean": 0.0,
                "mean_similarity_penalty": 0.0,
            }

        if "in_zpd" in trace_df.columns:
            zpd_alignment = float(trace_df["in_zpd"].mean())
        else:
            zpd_alignment = float(
                (
                    (trace_df["expected_p"] >= 0.55) & (trace_df["expected_p"] <= 0.75)
                ).mean()
            )
        similarity = float(trace_df["similarity_penalty"].mean())

        by_student = trace_df.groupby("StudentID")["Topic"].nunique()
        topic_coverage_mean = float(by_student.mean()) if len(by_student) > 0 else 0.0
        return {
            "zpd_alignment_rate": zpd_alignment,
            "topic_coverage_mean": topic_coverage_mean,
            "mean_similarity_penalty": similarity,
        }
