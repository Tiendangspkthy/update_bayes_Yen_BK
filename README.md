# pedago_ml
A research-oriented framework for adaptive question selection, combining machine learning, psychometric modeling, and educational theory to support both history-based recommendation and cold-start diagnostic question selection.

## Setup
### Requirements
* Python 3.11
* [uv](https://github.com/astral-sh/uv) - Modern Python package manager

### Install uv
If you don't have uv installed, you can install it using:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup the environment and python packages
```shell
make venv
```
This will:
- Install `uv` if not already installed
- Install the specified Python version
- Create a virtual environment in `.venv`
- Install the package with dev dependencies
- Set up pre-commit hooks

### Activate the virtual environment
```shell
source .venv/bin/activate
```

Or use `uv run` to execute commands in the virtual environment without activating it:
```shell
uv run python main.py
```


### Data
Get data from AEGlobal.

## Project Commands
### Quick start
#### Run pipeline
```shell
uv run python main.py
```

By default, the pipeline trains an AI model using the configuration in `pedago_ml_configs/model.yaml`.
To use a different setup (e.g., `xgboost.yaml`, `ncf.yaml`), update `main.py` to point to the desired config file.

The pipeline performs the following steps:

1. Load and preprocess response data
2. Split data into train/validation/test sets
3. Train the selected model or calibrate psychometric parameters
4. Estimate student-question suitability scores
5. Apply ZPD / Flow constraints
6. Rank or select next questions
7. Evaluate prediction, ranking, and pedagogical alignment metrics

### Run unit test
```shell
make test
```
Runs linting, type checking, and unit tests.

### Run styling
```shell
make style
```
Formats code with black, checks with flake8, and sorts imports with isort.

## Configuration

The project uses YAML configuration files located in `pedago_ml_configs/`:
- `edu_theory/zpd.yaml`
- `edu_theory/flow_theory.yaml`
- `ml_models/xgb.yaml`
- `ml_models/ncf.yaml`
- `ml_models/knowledge_tracing.yaml`
- `psychometrics/rasch.yaml`
- `psychometrics/posterior.yaml`
- `pipelines/history_based.yaml`
- `pipelines/cold_start.yaml`


### Architecture

The framework is designed for two adaptive question selection settings:

1. **History-based adaptive question selection**
   This setting assumes that a student already has historical response records. The system uses machine learning models such as XGBoost, Neural Collaborative Filtering, and Knowledge Tracing to estimate student-question interaction patterns. Educational theory modules such as Zone of Proximal Development and Flow Theory are then applied to adjust the recommendation score toward pedagogically suitable questions.

2. **Cold-start diagnostic question selection**
   This setting targets new students with little or no response history. The system first calibrates question difficulty using Rasch / IRT 1PL. A new student's ability is initialized with a prior distribution and updated after each response. The next question is selected by maximizing expected information while keeping the predicted probability of correctness within a pedagogically appropriate range.

The repository is organized around five core component groups:

- **features/**: Handles data loading, train/validation/test splitting, and feature engineering.
- **ml_models/**: Contains predictive models for history-based recommendation, including XGBoost, NCF, and Knowledge Tracing.
- **psychometrics/**: Contains Rasch / IRT-based item calibration, ability estimation, and posterior updating for cold-start diagnosis.
- **edu_theory/**: Implements pedagogical constraints and scoring mechanisms such as ZPD and Flow Theory.
- **pipelines/**: Provides task-specific pipelines for history-based recommendation and cold-start diagnostic selection.

The overall architecture is:

```text
Historical response data
        ↓
Feature engineering
        ↓
History-based ML / KT models
        ↓
ML-based correctness prediction
        ↓
ZPD / Flow scoring
        ↓
Hybrid question ranking
```
For the cold-start setting:
```text
Historical item responses
        ↓
Rasch / IRT item calibration
        ↓
Initialize student ability prior
        ↓
Observe response to selected question
        ↓
Update ability posterior
        ↓
Expected information + ZPD / Flow constraints
        ↓
Select next diagnostic question
```
After the cold-start phase, the estimated ability and uncertainty can be used to initialize the history-based adaptive practice pipeline.


## Evaluation

The framework evaluates both predictive performance and pedagogical suitability. Since the project focuses on adaptive question selection, the evaluation is organized into two groups: history-based evaluation and cold-start evaluation.

### 1. History-based Evaluation

The history-based pipeline evaluates how well the model ranks suitable next questions for students with existing response history.

Supported metrics include:

- **AUC / LogLoss**:
  Measures the quality of predicted correctness probabilities.

- **Accuracy / F1-score**:
  Evaluates binary correctness prediction when a classification threshold is used.

- **NDCG@K**:
  Evaluates whether highly suitable questions are ranked near the top.

- **Recall@K**:
  Measures whether relevant or suitable questions appear in the top-K recommendations.

- **HitRate@K**:
  Checks whether at least one relevant question appears in the top-K list.

- **MRR (Mean Reciprocal Rank)**:
  Measures how early the first relevant question appears in the ranked list.

- **ZPD Alignment Rate**:
  Measures the proportion of recommended questions whose predicted difficulty is within the target learning zone.

- **Flow Alignment Rate**:
  Measures whether recommended questions fall into the target probability-of-correctness range, avoiding questions that are too easy or too difficult.

- **Topic / Category Coverage**:
  Measures whether recommendations cover diverse topics or categories instead of repeatedly selecting similar questions.

### 2. Cold-start Evaluation

The cold-start pipeline evaluates how efficiently the system estimates a new student's ability and selects informative diagnostic questions.

Supported metrics include:

- **Theta Estimation Error**:
  Measures the error between estimated ability and reference ability when ground truth or simulated ability is available.

- **Posterior Uncertainty Reduction**:
  Measures how quickly uncertainty about student ability decreases after each diagnostic question.

- **Expected Information Gain**:
  Measures the amount of information gained from selected questions.

- **Number of Questions to Stable Ability Estimate**:
  Tracks how many diagnostic questions are needed before the ability estimate becomes stable.

- **Target Probability Alignment**:
  Measures whether selected questions fall within the desired probability-of-correctness range, such as 55%–75%.

- **ZPD / Flow Alignment**:
  Evaluates whether selected diagnostic questions are pedagogically appropriate, not only statistically informative.

- **Topic / Category Diversity**:
  Measures whether the diagnostic sequence avoids repeatedly selecting questions from the same topic or category.

- **Repeat Rate**:
  Ensures that the same question is not selected multiple times for the same student.

### 3. Ablation Evaluation

To support research analysis, the framework should compare multiple experimental settings:

- ML-only baseline
- ML + ZPD
- ML + Flow
- ML + ZPD + Flow
- Rasch / IRT-only cold-start
- Rasch / IRT + information-based selection
- Rasch / IRT + information + ZPD / Flow constraints
- Two-phase pipeline: cold-start diagnosis followed by history-based adaptive practice

The main goal of evaluation is to test whether combining machine learning, psychometric modeling, and educational theory improves adaptive question selection compared with model-only baselines.


## Project Structure

```
pedago_ml/
├── base/
│   ├── edu_theory.py         # base edu theory
│   ├── ml_model.py           # base ML model interface
│   ├── pipeline.py           # base pipeline interface
├── features/
│   ├── dataloader.py         # Data loader components
│   ├── splitter.py           # Split train/valid/test
│   └── feature_engineering.py  # Feature engineering components
├── edu_theory/
│   ├── zpd.py                # zone of proximal development implement
│   └── flow_theory.py        # flow theory implementation
├── ml_models/
│   ├── xgb.py                # xgb model
│   ├── ncf.py                # ncf model
│   └── knowledge_tracing.py  # knowledge tracing
├── psychometrics/
│   ├── rasch.py              # rasch model
│   ├── ability_estimator.py  # ability estimator
│   └── posterior_updater.py  # posterior updater
├── pipelines/
│   ├── history_based.py      # History based pipeline
│   └── cold_start.py         # Cold start pipeline
├── evaluation.py             # Evaluation pipeline
├── factory.py
└── scoring.py                # Scoring components

pedago_ml_configs/
├── edu_theory/
│   ├── zpd.yaml              # zone of proximal configuration
│   └── flow_theory.yaml      # flow theory configuration
├── ml_models/
│   ├── xgb.yaml              # xgboost configuration
│   ├── ncf.yaml              # neural collaborative filtering
│   └── knowledge_tracing.yaml  # knowledge tracing configuration
├── psychometrics/
│   ├── rasch.yaml            # rasch model configuration
│   └── posterior.yaml        # posterior updater configuration
├── pipelines/
│   ├── history_based.yaml    # History based pipeline configuration
│   └── cold_start.yaml       # cold-start pipeline configuration
```
