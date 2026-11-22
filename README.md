# ATP match predictor

Predicts the winner of an ATP match before it is played, from information known at the start of
the match (ranking, ranking points, age, height, hand, surface, tournament level, best-of).

Final project of the course "Machine learning in production". The model is trained with
scikit-learn, versioned data lives in DVC on Azure Blob Storage, runs and models are tracked in a
shared MLflow registry on DagsHub, and the web app serves the model selected by the `champion`
alias.

## Data source and license

Match data comes from [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp),
licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is used
here for non-commercial academic purposes only. Derived datasets keep the same license.
Details of each dataset version are in `ml/README_data.md`.

## Repository layout

```
.github/workflows/   CI (pull requests), CT, staging and deployment workflows
data/                DVC pointers for the prepared releases (v1.dvc, v2.dvc), data on Azure Blob
ml/                  data contract, data preparation, training pipeline, promotion policy
tests/               unit and pipeline tests, run in CI without DagsHub or Azure
app/                 web app and inference service (Dockerfile at the repository root)
```

## Branch flow

```
feature/* --PR--> dev --PR--> staging --PR--> main
```

- Create short-lived `feature/*` branches from `dev` and open a pull request into `dev`.
- Promote integrated work from `dev` to `staging`, then from `staging` to `main`, through pull
  requests only.
- `dev`, `staging` and `main` are protected: pull request required, 1 approval, the `ci` check must
  pass, no direct pushes, no admin bypass.
- Stacked work: a branch may start from the previous feature branch and target it; GitHub retargets
  the pull request to `dev` once the base branch is merged. Merge stacked pull requests in order.

## Workflows and triggers

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| CI | `ci.yml` | Pull request to `dev`, `staging` or `main` | `ruff check` and `pytest` (job `ci`, required check). Docker build and container smoke test to be added. |
| CT | `ct.yml` (planned) | `workflow_dispatch` on `main` | Fetch and validate a DVC dataset version, evaluate the champion, retrain, apply the promotion rule, move `champion`, or roll back. |
| Staging | planned | Push to `staging` | Build the image, start it with a test model and run the smoke test in Actions. |
| Deploy | planned | Push to `main` | Deploy the commit-tagged image to Azure Container Apps, wait for readiness, smoke test the public URL. |

## Interface contracts

These contracts are shared by the training pipeline, the web app and the deployment workflows.
Change them only through a pull request that updates every consumer.

### Prepared dataset

- One DVC-tracked release per version: `data/<version>/train.csv` and `data/<version>/evaluation.csv`,
  versions `v1` and `v2`.
- Columns, in this order (source of truth: `ml/contract.py`):

| Column | Type | Meaning |
|---|---|---|
| `match_id` | string | `<tourney_id>-<match_num>`, unique |
| `tourney_date` | int `YYYYMMDD` | Tournament start date, used to check chronology |
| `surface` | category | Court surface |
| `tourney_level` | category | Tournament level code from tennis_atp |
| `best_of` | int | 3 or 5 |
| `player_a_rank`, `player_b_rank` | int | ATP ranking before the match |
| `player_a_rank_points`, `player_b_rank_points` | int | ATP ranking points before the match |
| `player_a_age`, `player_b_age` | float | Age in years |
| `player_a_ht`, `player_b_ht` | int | Height in cm |
| `player_a_hand`, `player_b_hand` | category | Playing hand |
| `a_wins` | int | Label: 1 if player A won, 0 otherwise |

- Players A and B are assigned at random with a fixed seed, so the label is balanced and does not
  follow the `winner_*` / `loser_*` columns. Match statistics (aces, duration, score) are never used.
- No missing values: rows with a missing value are dropped during preparation, and validation
  rejects any missing value. Allowed category values and numeric ranges are defined in
  `ml/contract.py`.
- Every evaluation row is later than every training row (`tourney_date`).

### Model and registry

- Registered model: `atp-match-winner`. A separate name, `atp-match-winner-dev`, is used for local
  tests and controlled demonstrations.
- The production model is the version selected by the alias `champion`. Approved versions carry the
  tag `approval=approved`, plus `dataset_version` and `approval_roc_auc`.
- Input: a pandas DataFrame with the feature columns above (everything except `match_id`,
  `tourney_date` and `a_wins`). Output: `predict_proba(X)[:, 1]`, the probability that player A wins.
- Primary metric: ROC-AUC on the held-out evaluation split, higher is better. Log loss and accuracy
  are also logged.

### Web app endpoints

| Endpoint | Response |
|---|---|
| `GET /health` | 200 while the process is alive |
| `GET /ready` | 200 only once the champion is loaded and can predict, 503 before |
| `GET /model-info` | `model_name`, `model_version` (concrete version resolved from `champion` at startup), `run_id`, `dataset_version` |
| `POST /predict` | Body: one JSON object with the feature columns. Response: `probability_player_a_wins`, `predicted_winner` (`player_a` or `player_b`), `model_version`. Invalid input returns 422. |

### Configuration

Settings come from the environment, never from the code or the image.

| Name | Kind | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | GitHub variable | CT, app, deploy |
| `MLFLOW_TRACKING_USERNAME` | GitHub variable | CT, app, deploy |
| `MLFLOW_TRACKING_PASSWORD` | GitHub secret | CT, app, deploy |
| `AZURE_SAS_TOKEN` | GitHub secret | CT (DVC pull from Azure Blob) |
| `MODEL_NAME` | environment, default `atp-match-winner` | CT, app |
| `MLFLOW_EXPERIMENT_NAME` | environment, default `atp-match-winner` | CT |

Registered model names: `atp-match-winner` (production, used only by the CT workflow),
`atp-match-winner-dev` (local runs on the real data), `atp-match-winner-staging` (controlled
demonstrations on synthetic releases, see `ml/README_experiments.md`).

## Local development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Training pipeline against the shared registry (put the MLflow variables in a local `.env`, never
committed, and the SAS token in `.dvc/config.local`):

```bash
export MODEL_NAME=atp-match-winner-dev MLFLOW_EXPERIMENT_NAME=atp-match-winner-dev
python -m ml.pipeline --dataset-version v1   # writes results/decision.json
python -m ml.pipeline --rollback 1           # move champion back to an approved version
python -m ml.compare_models                  # model selection on training seasons only
```
