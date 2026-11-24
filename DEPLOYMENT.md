# Deployment

How the ATP match predictor is built, tested and deployed. The web app is a Docker image served
by Azure Container Apps over HTTPS. It loads the MLflow model selected by the `champion` alias
once, at startup.

## Infrastructure

| Item | Value |
|---|---|
| Cloud | Azure (Azure for Students subscription) |
| Resource group | `atp-predictor-rg`, region `francecentral` |
| Container Apps environment | `atp-env` |
| Container App | `atp-match-predictor`, external HTTPS ingress, 1 replica |
| Image registry | GitHub Container Registry (`ghcr.io/zorobabel33/atp-match-predictor`) |
| Image tag | Commit SHA (`${{ github.sha }}`), never `latest` |
| Model registry | MLflow on DagsHub |
| Data | DVC on Azure Blob Storage (used by CT only, not by the app) |

### How GitHub Actions authenticates to Azure

The tenant does not allow students to create app registrations, so a classic service principal
is not possible. Workflows use OIDC federation with a user-assigned managed identity
(`atp-gh-deploy-id`):

- The identity has the Contributor role on the resource group `atp-predictor-rg` only.
- Federated credentials trust this repository for the `main` and `staging` branches only.
- No password or key is stored anywhere. Jobs that call Azure need
  `permissions: id-token: write` and the `azure/login` action.

GitHub secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`MLFLOW_TRACKING_PASSWORD`, `AZURE_SAS_TOKEN`.
GitHub variables: `AZURE_RESOURCE_GROUP`, `AZURE_CONTAINERAPP_NAME`, `APP_URL`,
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`.

The DagsHub password is also stored as a Container App secret and exposed to the container as
`MLFLOW_TRACKING_PASSWORD` through a secret reference.

## Which branch triggers what

| Event | Workflow | Job | What happens |
|---|---|---|---|
| Pull request to `dev`, `staging` or `main` | `ci.yml` | `ci` | Lint, tests, Docker build and container smoke test |
| Push (merge) to `staging` | staging workflow | staging | Build the image tagged with the commit SHA, start it in Actions with the approved DagsHub staging champion, wait for `/ready`, run the smoke test |
| Push (merge) to `main` | deploy workflow | build, deploy | Build or select the verified image for the commit, deploy it to Azure Container Apps, wait for `/ready`, smoke test the public URL |
| Manual run from `main` | `ct.yml` | `train` | Run the training pipeline or a rollback |
| Manual run from `main`, when `champion` moved | `ct.yml` | `refresh-online-model` | Start a new revision, wait for `/ready`, check the loaded version with a live prediction |

Deployment jobs use a `concurrency` group so two deployments never overlap.

Deploy records the revision returned by the image update, then checks that exact revision
through `az containerapp revision show`. It must be active, provisioned, healthy and running,
with the expected commit-tagged image. The check does not require the app-level
`latestReadyRevisionName` field. The public HTTP smoke test remains a separate required step.

## First deployment and health probes

The initial Container App was created with the Azure quickstart image on port 80. Deploy
stores the DagsHub password as the `mlflow-password` Container App secret, switches the ingress
to port 8000 and replaces the quickstart image with the commit-tagged application image.
The GHCR package must be public so Azure can pull it without registry credentials.

Verify the following HTTP probes in Azure Portal after replacing the quickstart image:
Container Apps, `atp-match-predictor`, Containers, Edit and deploy, Health probes. Do not assume
that a Docker `HEALTHCHECK` configures Azure probes.

| Probe | Path | Port | Period | Failure threshold | Timeout |
|---|---|---|---|---|---|
| Startup | `/ready` | 8000 | 10 seconds | 30 | 5 seconds |
| Liveness | `/health` | 8000 | 10 seconds | 3 | 5 seconds |
| Readiness | `/ready` | 8000 | 10 seconds | 3 | 5 seconds |

Startup allows up to five minutes to load and test the champion. Readiness must use HTTP
`/ready`, which stays unavailable until a real prediction succeeds; a TCP port check alone
does not establish that the model is ready. Saving probes creates a new revision. Repeat the
public smoke test after it becomes healthy. These are required settings to verify in the
resource, rather than a claim that the initial quickstart configuration already included them.

If `azure/login` fails with `AADSTS700213`, compare the assertion subject in the failed job
with the managed identity's federated credentials. This repository presents subjects that
include the numeric owner and repository IDs. The identity must trust the exact subject for
`refs/heads/main` (and `staging` when needed), issuer
`https://token.actions.githubusercontent.com`, audience `api://AzureADTokenExchange`.
An Azure resource owner must make identity changes; no additional client secret is needed.

## Smoke test

Run against the staging container or the public production URL:

1. Poll `GET /ready` with a retry loop and a timeout until it returns 200. The app downloads the
   model from DagsHub at startup, so a fixed sleep is not reliable.
2. Call `GET /model-info` and read `model_version`.
3. Call `POST /predict` with a valid example and check that the response contains
   `probability_player_a_wins`, `predicted_winner` and a `model_version`.
4. Fail unless `model_version` equals the version expected from the `champion` alias
   (`ml.registry.get_champion`).

Example request body:

```json
{"surface": "Hard", "tourney_level": "G", "best_of": 5,
 "player_a_rank": 1, "player_a_rank_points": 9500, "player_a_age": 23.5, "player_a_ht": 188, "player_a_hand": "R",
 "player_b_rank": 45, "player_b_rank_points": 1200, "player_b_age": 29.1, "player_b_ht": 183, "player_b_hand": "L"}
```

## Reproduce a deployment

Prerequisites: Azure CLI with the `containerapp` extension, access to the resource group.

```bash
az login --tenant <TENANT_ID>

# Deploy a specific commit. Use the commit SHA tag, never latest.
az containerapp update \
  --name "$AZURE_CONTAINERAPP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --image ghcr.io/zorobabel33/atp-match-predictor:<COMMIT_SHA>

# Check the revisions and the loaded model.
az containerapp revision list -n "$AZURE_CONTAINERAPP_NAME" -g "$AZURE_RESOURCE_GROUP" -o table
curl -s "$APP_URL/model-info"
```

To run the same image locally:

```bash
docker build -t atp-match-predictor:local .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
  -e MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD" \
  -e MODEL_NAME=atp-match-winner \
  atp-match-predictor:local
```

## Changing the online model

The app resolves `champion` once at startup, so there is no hot reload. After a promotion or a
rollback, the CT job `refresh-online-model` starts a new revision, waits for `/ready`, then
checks that `/model-info` and a live `/predict` report the version stored in
`champion_version`. A rollback follows the same path and verifies the restored version.

### If the app does not become ready

Container Apps keeps serving the previous healthy revision while the new one is not ready, so
users are not affected. The workflow fails after its timeout. To recover:

1. Read the container logs of the failed revision (Log Analytics or `az containerapp logs show`).
2. Run the CT workflow with `operation=rollback` and `rollback_version` set to the previous
   approved version, so `champion` points to a model that is known to load.
3. Check that the public URL reports that version.

## Twelve-factor app

| Factor | How the project applies it |
|---|---|
| 1. Codebase | One Git repository. Staging and production are deploys of the same code, selected by branch (`staging`, `main`). |
| 2. Dependencies | Pinned in `requirements.txt` files and installed in the Dockerfile. Nothing is assumed from the host. |
| 3. Config | Environment variables only (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MODEL_NAME`, `PORT`). Secrets live in GitHub secrets and Container App secrets, never in the repository or the image. |
| 4. Backing services | MLflow on DagsHub and Azure Blob (DVC) are attached resources reached through URLs and credentials from the environment. Changing the tracking server needs no code change. |
| 5. Build, release, run | Build: one image per commit, tagged with the SHA. Release: the image plus its environment, as a Container Apps revision. Run: the revision. Releases can be rolled back by redeploying an earlier SHA. |
| 6. Processes | The app is stateless. The model is loaded into memory at startup, and nothing is written to local disk that matters. |
| 7. Port binding | The app is self-contained and listens on `PORT` (default 8000). Container Apps ingress provides HTTPS. |
| 8. Concurrency | Scaling is by adding replicas of the same process. It is fixed to one replica to stay within the student credit. Not a concern at this traffic. |
| 9. Disposability | Containers are replaceable. A new revision replaces the old one, the readiness probe keeps traffic on a healthy revision, and liveness uses `/health`. Startup is slower because the model is downloaded. |
| 10. Dev/prod parity | The same image runs in CI, staging and production, with the same Python version. The only differences are the model name and the credentials in the environment. |
| 11. Logs | The app writes to stdout. Container Apps sends it to Log Analytics. The app does not manage log files. |
| 12. Admin processes | Training, promotion and rollback are one-off runs of the CT workflow, not part of the web app. |

Factors that matter less here: factor 8 (a single replica is enough for a course demo) and
factor 6 beyond statelessness (there is no database or session state). The app has no local
state, so there is no sticky-session or data-persistence concern.
