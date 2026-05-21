from pathlib import Path

import numpy as np
import pandas as pd

from pedago_ml.features.splitter import ColdStartSplitter
from pedago_ml.psychometrics.ability_estimator import GridAbilityEstimator
from pedago_ml.psychometrics.posterior_updater import GridPosteriorUpdater
from pedago_ml.scoring import ColdStartSelector
from pedago_ml.utils.config import ColdStartPipelineConfig

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "pedago_ml_configs" / "cold_start.yaml"
)


def test_cold_start_config_loads():
    config = ColdStartPipelineConfig.load_config(CONFIG_PATH)
    assert config.rasch_config.model_name == "rasch"
    assert config.posterior_config.n_grid == 81
    assert config.selector_config.similarity.recent_window == 3
    assert "splitter" not in ColdStartPipelineConfig.model_fields


def test_cold_start_splitter_partitions_students():
    config = ColdStartPipelineConfig.load_config(CONFIG_PATH)
    splitter = ColdStartSplitter(config=config)
    df = pd.DataFrame(
        {
            "StudentID": [1, 1, 2, 2, 3, 3],
            "QuestionID": [10, 11, 10, 12, 11, 12],
            "Result": [1, 0, 1, 1, 0, 1],
        }
    )

    calibration_df, eval_df = splitter.run(df)
    calib_students = set(calibration_df["StudentID"].unique())
    eval_students = set(eval_df["StudentID"].unique())

    assert calib_students.isdisjoint(eval_students)
    assert calib_students | eval_students == {1, 2, 3}
    assert len(calib_students) == 2
    assert len(eval_students) == 1


def test_grid_posterior_updates_toward_higher_ability_after_correct_answer():
    config = ColdStartPipelineConfig.load_config(CONFIG_PATH)
    updater = GridPosteriorUpdater(config.posterior_config)
    estimator = GridAbilityEstimator(config.posterior_config)

    prior = updater.initialize()
    prior_summary = estimator.summarize(prior)

    posterior = updater.update(prior, beta_q=0.0, response=1)
    posterior_summary = estimator.summarize(posterior)

    assert posterior_summary.theta_mean > prior_summary.theta_mean
    assert posterior_summary.variance < prior_summary.variance


def test_selector_penalizes_recent_topic_repetition():
    config = ColdStartPipelineConfig.load_config(CONFIG_PATH)
    updater = GridPosteriorUpdater(config.posterior_config)
    selector = ColdStartSelector(config.selector_config)

    posterior = updater.initialize()
    asked = pd.DataFrame(
        [
            {"QuestionID": 101, "Topic": "number", "a_category": 1, "b_category": 0},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "QuestionID": 102,
                "item_difficulty": 0.0,
                "Topic": "number",
                "a_category": 1,
                "b_category": 0,
            },
            {
                "QuestionID": 103,
                "item_difficulty": 0.0,
                "Topic": "geometry",
                "a_category": 0,
                "b_category": 1,
            },
        ]
    )

    scored = selector.score_candidates(candidates, posterior, asked)
    score_same_topic = float(
        scored.loc[scored["QuestionID"] == 102, "final_score"].iloc[0]
    )
    score_other_topic = float(
        scored.loc[scored["QuestionID"] == 103, "final_score"].iloc[0]
    )

    assert np.isfinite(score_same_topic)
    assert np.isfinite(score_other_topic)
    assert score_other_topic > score_same_topic


def test_selector_constrained_zpd_filters_out_of_band_candidates():
    config = ColdStartPipelineConfig.load_config(CONFIG_PATH)
    config.selector_config.zpd.lower_probability = 0.55
    config.selector_config.zpd.upper_probability = 0.75
    selector = ColdStartSelector(config.selector_config)
    updater = GridPosteriorUpdater(config.posterior_config)

    posterior = updater.initialize()
    candidates = pd.DataFrame(
        [
            {
                "QuestionID": 201,
                "item_difficulty": -3.0,
                "Topic": "algebra",
            },
            {
                "QuestionID": 202,
                "item_difficulty": -0.5,
                "Topic": "algebra",
            },
        ]
    )

    scored = selector.score_candidates(candidates, posterior, pd.DataFrame())
    selected = int(scored.iloc[0]["QuestionID"])

    assert selected == 202
    assert bool(scored.loc[scored["QuestionID"] == 202, "in_zpd"].iloc[0])
    assert not np.isfinite(
        float(scored.loc[scored["QuestionID"] == 201, "final_score"].iloc[0])
    )
