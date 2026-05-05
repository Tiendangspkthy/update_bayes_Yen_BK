# pedago_ml

`pedago_ml` is a research-oriented framework for adaptive question selection. The project combines machine learning, psychometric modeling, and educational theory to study whether pedagogically constrained recommendation improves student-question selection compared with model-only baselines.

The repository currently focuses on a fixed research dataset for paper development: 3,850 students, each answering the same 30 questions. Because the dataset is fixed, the main engineering priority is reproducible experimentation rather than large-scale production ingestion.

## 1. Project Goals

The framework is designed to support two related adaptive learning settings:

1. **History-based adaptive question selection**
   - Uses existing student response history.
   - Trains predictive models such as XGBoost and Neural Collaborative Filtering.
   - Estimates the probability that a student will answer a candidate question correctly.
   - Applies educational-theory constraints such as Zone of Proximal Development (ZPD) and Flow Theory.
   - Ranks candidate questions for each student.

2. **Cold-start diagnostic question selection**
   - Targets new students with limited or no response history.
   - Uses psychometric modeling such as Rasch / IRT to estimate item difficulty and student ability.
   - Selects diagnostic questions using information gain while keeping the question within a suitable learning zone.

The current implementation is strongest for the history-based setting. Cold-start modules are part of the intended research architecture and can be expanded as the psychometric components mature.

## 2. Repository Structure

```text
pedago_ml/
├── base/
│   ├── edu_theory.py          # Base interface for educational-theory modules
│   ├── ml_model.py            # Base interface for ML models
│   └── pipeline.py            # Base interface for pipelines
├── features/
│   ├── dataloader.py          # Load interaction data and item metadata
│   ├── splitter.py            # Split train/validation/test by student history
│   └── feature_engineering.py # Build leakage-safe aggregate features
├── edu_theory/
│   ├── zpd.py                 # Zone of Proximal Development scoring
│   └── flow_theory.py         # Flow-theory scoring
├── ml_models/
│   ├── xgb.py                 # XGBoost correctness prediction model
│   ├── ncf.py                 # Neural Collaborative Filtering model
│   └── knowledge_tracing.py   # Knowledge tracing model
├── psychometrics/
│   ├── rasch.py               # Rasch / IRT item calibration
│   ├── ability_estimator.py   # Student ability estimation
│   └── posterior_updater.py   # Ability posterior update logic
├── pipelines/
│   ├── history_based.py       # History-based adaptive recommendation pipeline
│   └── cold_start.py          # Cold-start diagnostic pipeline
├── utils/
│   ├── config.py              # Pydantic configuration classes
│   └── logger.py              # Shared logging utility
├── evaluation.py              # Evaluation logic
├── factory.py                 # Component factory
└── scoring.py                 # Scoring and ranking utilities

pedago_ml_configs/
├── history_based.yaml         # Main config for history-based experiments
└── cold_start.yaml            # Main config for cold-start experiments
```

Configuration is intentionally kept flat under `pedago_ml_configs/`. Model-specific and educational-theory settings are nested inside the main pipeline config instead of being stored in separate subfolders.

## 3. Data Format

### 3.1 Interaction data

The interaction file is configured through:

```yaml
interaction_data_path: "data/full_data.csv"
```

Expected format:

```text
StudentID,1,2,3,...,30
10001,1,0,1,...,0
10002,0,1,1,...,1
```

Each row is one student. Each question column stores the binary response result:

- `1`: correct answer
- `0`: incorrect answer

The dataloader melts this wide table into interaction-level format:

```text
StudentID | QuestionID | Result
```

### 3.2 Item metadata

The item metadata file is configured through:

```yaml
item_data_path: "data/item_specification.csv"
```

Expected core columns:

```text
ItemID | Specification
```

The `Specification` field is parsed into item-level features:

- `Level`
- `Topic`
- category indicator columns derived from the remaining specification tokens

These item features are merged into the interaction-level dataset by `QuestionID`.

## 4. Installation

### Requirements

- Python 3.11
- `uv`

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment

```bash
make venv
```

This command creates the virtual environment, installs dependencies, and prepares the development setup.

### Activate environment

```bash
source .venv/bin/activate
```

Or run commands directly with:

```bash
uv run python main.py
```

## 5. Configuration

The main history-based configuration is:

```text
pedago_ml_configs/history_based.yaml
```

It contains four major groups:

```yaml
interaction_data_path: "data/full_data.csv"
item_data_path: "data/item_specification.csv"

user_id: "StudentID"
item_id: "QuestionID"
target: "Result"

item_features:
  - name: "Level"
    type: "categorical"
  - name: "Topic"
    type: "categorical"

splitter:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1

smoothing: 5

xgb_config:
  ...

ncf_config:
  ...

zpd:
  ...
```

## 6. History-Based Pipeline

The history-based pipeline follows this flow:

```text
Raw interaction data
        ↓
Dataloader
        ↓
Student-level train/validation/test split
        ↓
Leakage-safe feature engineering
        ↓
XGBoost / NCF correctness prediction
        ↓
ZPD / Flow pedagogical adjustment
        ↓
Per-student candidate ranking
        ↓
Evaluation
```

### Run pipeline

```bash
uv run python main.py
```

The default entrypoint should load:

```python
from pedago_ml.utils.config import HistoryBasedPipelineConfig

config = HistoryBasedPipelineConfig.load_config(
    "pedago_ml_configs/history_based.yaml"
)
```

## 7. Modeling Logic

### 7.1 Prediction objective

The main supervised learning target is:

```text
P(Result = 1 | StudentID, QuestionID, student history features, item features)
```

This probability is used as the base suitability signal.

### 7.2 Pedagogical adjustment

Model probability alone may recommend questions that are too easy. ZPD and Flow modules adjust the score toward questions that are neither too easy nor too difficult.

A typical ZPD target range is:

```text
0.50 <= predicted probability <= 0.80
```

The ideal target can be set around:

```text
target_probability = 0.65
```

This encourages questions with moderate challenge, where the student is likely to learn rather than simply repeat already-mastered items.

### 7.3 Final ranking

For each student, candidate questions are ranked by adjusted score:

```text
zpd_adjusted_score = weighted_average(base_model_score, zpd_score)
```

The top-ranked items are treated as recommended next questions.

## 8. Evaluation

The project evaluates both predictive quality and pedagogical suitability.

### 8.1 Predictive metrics

- **LogLoss**: probability calibration and prediction quality
- **AUC**: ranking separation between correct and incorrect responses
- **Accuracy**: binary prediction quality under a threshold
- **F1-score**: classification balance under a threshold

### 8.2 Ranking metrics

- **NDCG@K**: whether useful items are ranked near the top
- **Recall@K**: whether relevant items are included in top-K
- **HitRate@K**: whether at least one relevant item appears in top-K
- **MRR**: how early the first relevant item appears

### 8.3 Pedagogical metrics

- **ZPD Alignment Rate**: proportion of recommendations inside the ZPD range
- **Flow Alignment Rate**: proportion of recommendations inside the desired probability range
- **Topic / Category Coverage**: diversity of recommended content
- **Repeat Rate**: whether the same question is repeatedly recommended to the same student

## 9. Ablation Design

Because this is a research framework, experiments should compare model-only and pedagogically adjusted variants.

Recommended ablation settings:

| Experiment | Description |
|---|---|
| ML-only | Rank by base predicted probability only |
| ML + ZPD | Adjust ranking using ZPD score |
| ML + Flow | Adjust ranking using Flow score |
| ML + ZPD + Flow | Combine ML score with multiple educational-theory constraints |
| Rasch-only | Select questions using psychometric difficulty only |
| Rasch + information gain | Select questions using expected information |
| Rasch + information gain + ZPD | Combine diagnostic informativeness with pedagogical suitability |

The key research question is whether educational-theory-aware ranking improves adaptive question selection compared with pure model-based ranking.

## 10. Development Commands

### Run tests

```bash
make test
```

### Format and style

```bash
make style
```

### Run a single module manually

```bash
uv run python -m pedago_ml.ml_models.xgb
uv run python -m pedago_ml.ml_models.ncf
```

If module-level scripts still reference old nested config paths such as `pedago_ml_configs/pipelines/history_based.yaml`, update them to:

```text
pedago_ml_configs/history_based.yaml
```

## 11. Design Notes

- The dataset is fixed, so reproducibility and leakage-safe evaluation are more important than general ingestion flexibility.
- The history-based splitter should not be interpreted as cold-start student generalization; it splits each student's answered questions into observed and hidden portions.
- Leave-one-out features are used on train rows to reduce target leakage.
- Validation and test rows only use statistics computed from train data.
- ZPD / Flow modules should be treated as pedagogical reranking layers, not replacements for the base predictive model.
- Model-only baselines must always be preserved for ablation comparison.
