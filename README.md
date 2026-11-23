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
| CT | `ct.yml` | Manual `workflow_dispatch` from `main` (Actions tab, available once the file is on `main`). Inputs: `operation` (`pipeline` or `rollback`), `dataset_version` (`v1`, `v2`), `rollback_version`. | Job `train`: `dvc pull` the version from Azure, validate it, score the champion, retrain when the policy says so, promote or keep the champion, check that the champion predicts, upload `decision.json`. Rollback moves `champion` to a previously approved version. Job `refresh-online-model` runs only when `champion` moved: placeholder to be replaced by the redeploy and live verification. Runs are serialized (`concurrency`). |
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

### Web app

The Courtside interface compares two ATP player profiles, lets users edit all 13 input features,
and shows the predicted winner, each player's probability and the loaded model version. It is
served by FastAPI at `/`, with no JavaScript build step or external request from the browser.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Export MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD locally.
# For the real-data development model, also export MODEL_NAME=atp-match-winner-dev.
python -m app.main  # listens on PORT, default 8000
```

Open `http://localhost:8000`. The page and player catalogue remain accessible if the model cannot
load; `/ready`, `/model-info` and valid prediction requests return 503. Startup resolves the
approved `champion` once, loads that immutable version and runs a probe before becoming ready.
Restart the service to pick up a changed alias. `/health` remains 200 while loading or unavailable.
`/contract` provides the shared field names, categories and bounds to the form.

For an entirely offline UI preview after installing dependencies:

```bash
python -m scripts.create_demo_model
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/results/demo/mlflow.db"
export MODEL_NAME=atp-match-winner-ui-demo
python -m app.main
```

This creates a local approved model trained on synthetic fixtures. The UI identifies it as a
demonstration, and its probabilities are illustrative. This is an explicit development tool,
never a fallback for an unavailable production model. No DagsHub or Azure credentials are needed.
The registry and artifacts live in the ignored `results/demo/` directory.

Example prediction (names and player IDs are not sent to the model):

```bash
curl http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"surface":"Hard","tourney_level":"M","best_of":3,
       "player_a_rank":2,"player_a_rank_points":9960,"player_a_age":23.1,
       "player_a_ht":183,"player_a_hand":"R",
       "player_b_rank":7,"player_b_rank_points":3760,"player_b_age":39,
       "player_b_ht":188,"player_b_hand":"R"}'
```

Response shape (illustrative values; the probability depends on the loaded model):

```json
{"probability_player_a_wins": 0.62, "predicted_winner": "player_a", "model_version": "1"}
```

Invalid input returns 422: missing or extra fields, wrong types, unsupported categories,
out-of-range or non-finite numbers. Tests cover valid predictions, invalid requests, startup
readiness, a newer registered version, a moved alias and unavailable or unapproved champions,
using only synthetic data and a temporary SQLite registry.

### Player catalogue

`app/static/players.json` contains 200 complete ranked player profiles. It is a dated snapshot
(2026-06-08), not live rankings. Age is calculated at the ranking date. Unknown height, birth date
or hand, and values outside `ml.contract`, are excluded without imputation. Fields can be edited
in the UI; a free profile is also available. Selecting the same named player on both sides is
rejected by the interface.

Regenerate it with `python -m scripts.build_players`. The script downloads `atp_players.csv` and
`atp_rankings_current.csv` from the same pinned archive commit as `ml.prepare_data`, caches the
raw files in ignored `raw/players/`, joins by player ID, selects the newest ranking date and
writes the first 200 complete profiles. `--raw-dir`, `--output` and `--limit` are configurable.
Network is used only during explicit generation, never at app startup, in the UI or in CI tests.
Source commit and SHA-256 hashes are recorded inside the JSON. The derived catalogue shares
the source's CC BY-NC-SA 4.0 license; attribution and snapshot date are visible in the UI.

### Training and checks

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
