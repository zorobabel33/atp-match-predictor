# ATP match predictor

Predicts the winner of an ATP match before it is played, from information known at the start of
the match (ranking, ranking points, age, height, hand, surface, tournament level, best-of).

Final project of the course "Machine learning in production". The model is trained with
scikit-learn, versioned data lives in DVC on Azure Blob Storage, runs and models are tracked in a
shared MLflow registry on DagsHub, and the web app serves the model selected by the `champion`
alias.

Live demo: [Courtside on Azure](https://atp-match-predictor.happybush-1050e9f8.francecentral.azurecontainerapps.io/).
See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment checks and recovery.

## Try it

Open the URL above, pick two players, a surface, a tournament level and a match format, then
predict. The page shows the probability for each player and the model version that answered.

The same service as an API (replace `$APP_URL` with the URL above):

```bash
curl $APP_URL/model-info
# {"model_name":"atp-match-winner","model_version":"1","run_id":"e416799e01dc48c0909ab7f5aa1c7c9b","dataset_version":"v1","data_source":"dvc"}

curl -X POST $APP_URL/predict -H 'Content-Type: application/json' -d '{
  "surface": "Hard", "tourney_level": "G", "best_of": 5,
  "player_a_rank": 1, "player_a_rank_points": 9500, "player_a_age": 23.5, "player_a_ht": 188, "player_a_hand": "R",
  "player_b_rank": 45, "player_b_rank_points": 1200, "player_b_age": 29.1, "player_b_ht": 183, "player_b_hand": "L"}'
# {"probability_player_a_wins":0.9462046235037788,"predicted_winner":"player_a","model_version":"1"}
```

Sample inputs and the answers of the production model (version 1, 2026-11-24):

| Match | Player A | Player B | P(A wins) | Predicted winner |
|---|---|---|---|---|
| Grand Slam, hard, best of 5 | rank 1, 9500 pts, 23.5 y, 188 cm, R | rank 45, 1200 pts, 29.1 y, 183 cm, L | 0.946 | `player_a` |
| Masters, clay, best of 3 | rank 12, 2800 pts, 27.0 y, 185 cm, L | rank 3, 7000 pts, 24.2 y, 191 cm, R | 0.219 | `player_b` |
| ATP 250/500, grass, best of 3 | rank 30, 1500 pts, 25.0 y, 196 cm, R | rank 28, 1550 pts, 33.0 y, 180 cm, R | 0.594 | `player_a` |

Values outside the contract are rejected with HTTP 422, for example `"surface": "Carpet"` or
`"player_a_rank": 0`. `GET /health` and `GET /ready` report liveness and readiness.

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
| CI | `ci.yml` | Pull request to `dev`, `staging` or `main` | Required job `ci`: `ruff check`, `pytest`, Linux amd64 Docker build, start the image with an approved local test champion, wait for readiness, verify model metadata, a real prediction and invalid input. No DagsHub or Azure credentials. |
| CT | `ct.yml` | Manual `workflow_dispatch` from `main` (Actions tab, available once the file is on `main`). Inputs: `operation` (`pipeline` or `rollback`), `dataset_version` (`v1`, `v2`), `rollback_version`. | Job `train`: `dvc pull` the version from Azure, validate it, score the champion, retrain when the policy says so, promote or keep the champion, check that the champion predicts, upload `decision.json`. Rollback moves `champion` to a previously approved version. Job `refresh-online-model` runs only when `champion` moved: it starts a new Container App revision, waits until the public URL serves the expected version, then checks it with a live prediction. Runs are serialized (`concurrency`). |
| Staging | `staging.yml` | Push to `staging` | Build the image from the merged commit, start it against the approved `atp-match-winner-staging` champion on DagsHub and run the smoke test: readiness, model metadata, a real prediction, invalid input. |
| Deploy | `deploy.yml` | Push to `main` | Job `build`: build the image, smoke test it with a local test champion, push it to GHCR tagged with the commit SHA. Job `deploy`: update the Container App to that image, wait for the new revision to be ready, then smoke test the public URL against the expected production champion (real prediction and model version). |

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

- Registered model: `atp-match-winner`. Separate names are used for local runs on the real data
  (`atp-match-winner-dev`) and for controlled demonstrations (`atp-match-winner-staging`).
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

### Docker

Build the production image from the repository root:

```bash
docker build -t atp-app .
```

The image uses Python 3.12 slim, installs only `app/requirements.txt`, and runs as UID/GID 10001.
MLflow and scikit-learn match training. SQLAlchemy is pinned to the training version too:
MLflow 2.22.5's SQLite store imports a pool class removed in SQLAlchemy 2.1.
The allowlist in `.dockerignore` sends only the app, static player catalogue and four inference
modules to the build. Training datasets, tests, notebooks, DVC, local registries, development
files and `.env` files are excluded. The model is downloaded at startup, never during the build.

Put the real registry settings in a local, ignored `.env` (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`; optionally `MODEL_NAME`). Then run:

```bash
docker run --rm --name atp-app --publish 127.0.0.1:8000:8000 --env-file .env atp-app
```

Open `http://localhost:8000`. To test the shared real-data development model instead of the
production default, add `--env MODEL_NAME=atp-match-winner-dev` before the image name.
For deployment, inject these settings using the hosting platform's runtime configuration.
Do not pass credentials as Docker build arguments. `PORT` defaults to 8000; if changed, publish
the matching container port. The Docker healthcheck uses `/ready` and the configured `PORT`.
A failed champion load leaves the process alive but not ready, with no demo fallback. Fix the
runtime settings or approved registry version and restart the container.

On Apple Silicon the default local build is Linux arm64. Build with `--platform linux/amd64`
when validating the same architecture as CI and the intended deployment image. Images in CI
are tagged with the Git commit SHA; CI does not push an image or deploy a service.

### Reproduce the Docker CI check

After installing `requirements-dev.txt`, run the same checks locally (port 18000 keeps the UI
development server on 8000 available):

```bash
ruff check .
pytest
docker build --platform linux/amd64 -t atp-app:ci .
# Use a new, empty output directory for each test registry.
python -m tests.make_test_registry --output-dir results/container-registry
docker run --detach --name atp-ci --publish 127.0.0.1:18000:8080 \
  --mount "type=bind,source=$(pwd)/results/container-registry,target=/registry" \
  --env MLFLOW_TRACKING_URI=sqlite:////registry/mlflow.db \
  --env MODEL_NAME=atp-match-winner-ci --env PORT=8080 atp-app:ci
python -m scripts.smoke_test --url http://127.0.0.1:18000 \
  --expected-model-info results/container-registry/expected-model.json --timeout 90
docker logs atp-ci
docker rm --force atp-ci
```

The fixture contains two approved versions and deliberately keeps `champion` on version 1.
Artifact references use `/registry/model`, matching the container mount rather than the host's
directory. MLflow 2 writes a `registered_model_meta` file when loading a local model, so the
disposable synthetic model directory is writable by the container; the other fixture files
remain readable. This fixture is never included in the production image.
The smoke command waits for readiness, checks health and the HTML page,
compares all four model metadata fields to independently generated expectations, submits the
13-feature prediction and verifies its probability, winner and version. A rank of zero must
return 422. Unexpected responses or a readiness timeout return a nonzero exit code.

The Docker steps are in the existing required `ci` job, so a build, startup or prediction failure
fails the same branch-protection check. The workflow prints container logs on failure and removes
the test container even if a step fails. Dependencies and the base image are downloaded during
setup/build; fixture generation and application checks use no external MLflow or Azure service.

Local verification on 2026-11-23: Linux amd64 build succeeded, Docker reported the image as
1.05 GB, the container became healthy on `PORT=8080`, and the smoke test verified champion 1
while approved version 2 existed. A disposable container with a deliberately broken `/predict`
still passed readiness and model-info, then returned HTTP 500 and made the smoke command exit 1.
`ruff check .` and all 138 tests passed. Shared DagsHub serving and a GitHub-hosted CI run still
need validation with the team's runtime settings and a pull request, respectively.

### Web app

The Courtside interface compares two players selected from the ATP catalogue. Ranking, points,
age, height and hand are fixed to each selected player's snapshot; only surface, tournament
level and match format can be changed. It shows the predicted winner, each player's probability
and the loaded model version. FastAPI serves it at `/`, with no JavaScript build step or external
request from the browser.

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
or hand, and values outside `ml.contract`, are excluded without imputation. The interface displays
player statistics as non-editable values and builds predictions from the selected catalogue
profiles. Selecting the same player on both sides is rejected by the interface.

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

## Use of AI tools

We used AI coding assistants (Claude Code and ChatGPT) to work faster. Each member is the author
of their part: we made the design decisions and reviewed and tested the code.

### Enock: data, training pipeline, continuous training, team handoff

My work on the project:

- Problem framing and leakage rules: pre-match features only, a seeded random player order
  instead of the `winner_*` / `loser_*` columns, a chronological v1 / v2 split.
- Data: when the original `tennis_atp` repository disappeared, the switch to a pinned archive
  under the same license; the filters (Davis Cup, walkovers, missing values); versioning with DVC
  on the course Azure container, tags `data-v1` and `data-v2`.
- Model and quality gates, calibrated on the 2021 validation season before any evaluation on 2022
  or 2023. When the real v2 candidate did not improve (-0.0006), I kept the gates and demonstrated
  promotion and rollback on a separate staging model.
- Training pipeline, MLflow registry with the `champion` alias, and the CT workflow with rollback.
- Repository and team organisation: branch protections, stacked pull requests, interface
  contracts, the issues and briefs that let Paul and Adria work in parallel, and the reviews of
  their pull requests.

The assistant drafted code, tests and documentation from my specifications and ran routine checks,
which let me focus on these decisions and on verification. I approved every change and verified
the results myself (CI, `dvc pull` from a fresh clone, MLflow runs on DagsHub). My commits made
before 2026-11-23 carry an automatic `Co-Authored-By: Claude` trailer, turned off afterwards.

### Paul: web app, application tests, Docker, CI

My work on the project:

- The Courtside interface: a sober tennis-inspired design with the two players side by side,
  searchable player selection, fixed player statistics, match settings and a probability comparison.
  The 200-player catalogue comes from the same pinned archive as the training data, with its
  snapshot date and attribution displayed, so the interface needs no live player API.
- The inference API: input validation derived from `ml/contract.py`, the agreed `/predict`,
  `/health`, `/ready` and `/model-info` endpoints, and the approved `champion` resolved once at
  startup to a concrete version. Readiness requires a successful model prediction.
- Application tests on a temporary local MLflow registry: prediction format, malformed and
  out-of-range inputs, unavailable or unapproved models, and the original champion remaining
  loaded when a newer approved version exists or the alias changes. The integrated project
  suite passed 141 tests without DagsHub or Azure credentials.
- The production Docker image: Python 3.12 slim, serving dependencies only, a non-root user and
  an allowlist `.dockerignore`. The model is fetched at runtime, with credentials and training
  datasets kept out of the image.
- The required `ci` job: Linux amd64 Docker build, container startup with a synthetic approved
  champion, a readiness wait and HTTP smoke checks for model metadata, a real prediction and
  invalid-input rejection. A disposable container with a deliberately broken `/predict`
  confirmed that the smoke check fails on a regression.

ChatGPT helped generate the implementation, tests and Docker/CI configuration, split the work
into smaller tasks, troubleshoot integration and explain the development workflow.

### Adria: deployment

To be completed by Adria.
