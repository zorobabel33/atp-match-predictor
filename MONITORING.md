# Courtside: logging, monitoring and alerting

This adapts the monitoring exercise to the existing ATP predictor. It extends the same FastAPI
app and Docker image; the web UI, pinned champion startup, readiness check and 13-feature API
contract are preserved. The default exercise runs locally and never changes the Azure app or
the production champion.

## Start the complete offline exercise

Requirements: Docker Desktop running and the project's Python 3.12 environment active.

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.monitoring_demo
docker compose -f compose.yml -f monitoring/compose.demo.yml up --build -d
```

Run `monitoring_demo` once in a fresh checkout. If it says the registry already exists, reuse it;
do not delete it during the exercise. It creates two approved synthetic models in
`results/monitoring-demo`, both with their own training profile, and selects version 1. Paths in
this disposable registry are for `/registry` inside Docker. Its writable permissions are only
for the synthetic fixture, never for production models or credentials.

Open these local URLs:

| Service | URL | Purpose |
|---|---|---|
| Courtside | http://localhost:18000 | Same player comparison UI, synthetic demonstration model |
| API readiness | http://localhost:18000/ready | 200 only after a real startup prediction succeeds |
| Metrics | http://localhost:18000/metrics | Prometheus text exposition |
| Prometheus targets | http://localhost:9090/targets | `api` must be UP |
| Grafana | http://localhost:3000 | Log in with the exercise defaults `admin` / `admin`; skip the password change for the local demo |
| Mailpit inbox | http://localhost:8025 | Receives alert emails locally; no SMTP account needed |

Grafana provisions the Prometheus datasource, the **Courtside · ATP API** dashboard and all three
email alerts automatically. Look under Dashboards > Courtside, and Alerting > Alert rules.
The six panels show request rate, 5xx error percentage, p95 latency, predictions by class and
model, feature drift and the current model version. Request panels filter on `/predict` so
Prometheus scrapes and Azure health checks do not inflate the prediction graphs.

Run the repeatable verification (the timed alert test takes around ten minutes):

```bash
python -m scripts.monitoring_check
python -m scripts.monitoring_check --alerts --rollback
```

It checks the running stack, baseline drift, real firing/resolved emails in Mailpit and model
versions 1 -> 2 -> 1, then writes `.exercise-results/check.json`. It refuses a non-demo model;
failing availability scenarios restore the API in a `finally` block.

All ports bind to loopback only. Prometheus and Grafana retain their history in Docker volumes.
API logs rotate at three files of 10 MB. The API keeps its bounded drift window in memory;
restarting it resets that window. This stack assumes one Uvicorn process/replica, matching the
course deployment. A production fleet needs a shared or per-replica monitoring design.

## Differences from the bank example

| Exercise assumption | ATP adaptation |
|---|---|
| A separate `api/app.py` and Dockerfile | Extend `app/main.py` and the existing allowlist Dockerfile |
| 19 bank inputs and `y` | 13 tennis inputs; `a_wins`, `match_id` and dates are removed by replay |
| Class 1 / class 0 | Player A wins / player B wins; the API response fields remain unchanged |
| DVC v1, v2, v3 | This repository has real DVC v1 and v2 only. Demo v3 is explicitly synthetic shifted traffic, not a third real release |
| `compare_distributions` and a training profile already exist | Added shared PSI calculation, training-profile artifacts and CT drift reports |
| Missing input causes 5xx | Our strict contract correctly returns 422. Use an unavailable demo model for the 5xx alert |
| A Gmail/Mailtrap SMTP account | Mailpit captures real SMTP messages locally; an external provider is optional |

## Structured logs and request IDs

`LOG_LEVEL` controls app logs, which are JSON objects on stdout. Uvicorn access logs are disabled;
Uvicorn/MLflow lifecycle lines may still be plain text, as allowed by the exercise.

Each request returns `X-Request-ID`, accepting an incoming safe ID of at most 64 characters or
generating a UUID. Its request log and prediction log carry the same ID, including in FastAPI's
worker thread. Logs contain method, normalized route, status, duration, model version, predicted
class, probability, latency, feature count and missing count. Invalid requests are error-level
logs; a successful strict-contract prediction has 13 features and zero missing values. No input
row, authorization header, query string, signed URL or exception detail is logged. Unknown
paths share `/unmatched`, preventing unbounded metric labels.

```bash
# Send normal traffic from this champion's own training data.
python -m scripts.replay --file .exercise-data/v1/train.csv --shuffle --limit 1000

# Send five malformed requests: expect 422, not 500.
python -m scripts.replay --file .exercise-data/v1/train.csv --broken --limit 5

# Only errors.
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.level == "error")'

# Follow one request. Replace the example with an X-Request-ID from a response or log.
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.request_id == "duel-123")'

# Count prediction log records per model version.
docker compose logs api --no-log-prefix | jq -R -s 'map(fromjson? | select(.event == "prediction_made")) | group_by(.model_version) | map({model_version: .[0].model_version, predictions: length})'
```

Compose commands that only read logs can omit the demo override; commands that create/recreate
containers must include it. Always use `-f compose.yml -f monitoring/compose.demo.yml` for demo
startup to avoid accidentally selecting a DagsHub model.

## Metrics and the reference distribution

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | method, path, status |
| `http_request_duration_seconds` | Histogram | path |
| `predictions_total` | Counter | prediction (`0` or `1`), model_version |
| `model_info` | Gauge = 1 for the loaded model | model_version |
| `feature_drift` | Gauge, PSI for each feature | feature |

`drift_reference_ready` distinguishes an unavailable reference from a warm-up;
`drift_window_size` reports observations accumulated. `feature_drift_threshold` is 0.20.
Drift is first computed after 500 successful predictions, then every 100, using the most recent
500 observations. Failed requests and startup probes never enter that window. Metrics are
isolated per app process, so restarts and tests do not duplicate collectors.

The reference is `training-profile.json` in the immutable model's **approval run**, selected by
its `approval_run_id` tag, rather than the current alias or the latest experiment. Model run ID
and dataset version must match. Numeric bins are fixed training deciles; categorical bins are
the contract values plus an unseen-value bucket. PSI uses normalized frequencies with 1e-4
smoothing. The same `compare_distributions` function and `DRIFT_THRESHOLD` are used by CT's
`drift.json` and online monitoring. Drift signals investigation, not automatic retraining;
existing ROC-AUC promotion gates remain unchanged.

To display drift:

```bash
python -m scripts.replay --file .exercise-data/v1/train.csv --shuffle --limit 1000
python -m scripts.replay --file .exercise-data/v3/evaluation.csv --limit 868 --rate 5
```

Demo v3 forces grass, a top-ranked player A and a left-handed player A. Those feature curves
rise. They are a controlled scenario, not a claim about recent ATP data. Replay 1,000 shuffled
v1 training rows afterwards to restore the reference distribution.

## Email rules and scenarios

Mailpit is the default SMTP server. Open http://localhost:8025 to see both firing and resolved
messages addressed to `paul@courtside.test`. No message leaves the local stack. Provisioned rules:

| Rule | Condition | Pending duration | No-data state |
|---|---|---|---|
| API down | Prometheus `up{job="api"}` = 0 | 1 minute | Alerting |
| High error rate | More than 5% of `/predict` requests are 5xx over 2 minutes | 2 minutes | Normal (`OK`) |
| Data drift | Any feature PSI >= 0.20 | 2 minutes | Normal (`OK`) |

Evaluation runs every 10 seconds; grouping adds 10-30 seconds. Error-rate recovery can take a
further two-minute rate window. Wait for firing and recovery messages, not just a green page.

**API down:**

```bash
docker compose -f compose.yml -f monitoring/compose.demo.yml stop api
# Wait for the firing message in Mailpit, then:
docker compose -f compose.yml -f monitoring/compose.demo.yml start api
# Wait for /ready and the resolved message.
```

**High error rate:** select a nonexistent model only in the isolated local registry. The API
stays up and exports metrics, while valid inference requests correctly return 503.

```bash
DEMO_MODEL_NAME=missing-monitoring-demo docker compose -f compose.yml -f monitoring/compose.demo.yml up -d --no-deps api
python -m scripts.replay --file .exercise-data/v1/train.csv --limit 1800 --rate 10
# Restore the real demo champion, then normal traffic and wait for resolution.
docker compose -f compose.yml -f monitoring/compose.demo.yml up -d --no-deps api
python -m scripts.replay --file .exercise-data/v1/train.csv --shuffle --limit 1000
```

**Drift:** replay the synthetic v3 command above after normal traffic. Wait for the firing email,
then replay normal traffic again and wait for its resolved email. The alert explicitly says to
investigate and not automatically retrain.

For an external SMTP provider later, copy `.env.monitoring.example` to `.env` without overwriting
existing secrets, configure its SMTP settings and recipient, and use MandatoryStartTLS where
required. Restart Grafana. Never commit `.env`. The demo override always pins SMTP to Mailpit.

## Roll back and watch

Only this synthetic local registry is modified:

```bash
python -m scripts.monitoring_demo --switch-version 2
docker compose -f compose.yml -f monitoring/compose.demo.yml restart api
curl http://localhost:18000/model-info
python -m scripts.replay --file .exercise-data/v2/train.csv --shuffle --limit 100

python -m scripts.monitoring_demo --switch-version 1
docker compose -f compose.yml -f monitoring/compose.demo.yml restart api
python -m scripts.replay --file .exercise-data/v1/train.csv --shuffle --limit 1000
```

Wait for `/ready` after each restart. Grafana's version panel and prediction series change;
`model_loaded`, `drift_reference_loaded` and `prediction_made` logs identify each version.
The new process loads that version's training profile and starts a fresh drift window. A real
DagsHub rollback uses the existing CT workflow and must select another approved version;
do not substitute the demo registry or bypass production approval gates.

## Use the real DagsHub champion and Azure

New CT training runs now log `training-profile.json`. Older approved versions do not have that
artifact. They can still serve; `drift_reference_ready=0` and a warning identify the missing
monitoring reference, with no invented zero drift or inference outage.

After configuring existing MLflow and Azure DVC credentials locally, backfill the original
champion reference from its exact hash-verified DVC training release:

```bash
# Export the existing MLFLOW_TRACKING_* and Azure DVC credentials from your ignored .env.
python -m scripts.monitoring_data --backfill-reference
docker compose up --build -d
```

This adds only a missing profile artifact to the original approval run; it never overwrites a
profile, retrains a model or moves the champion alias. Then restart the API. Normal traffic must
come from `.exercise-data/real/<champion dataset_version>/train.csv`, shuffled. Real downloads
are kept separate from the synthetic demo CSVs. Backfill each legacy
approved version used in a real rollback, selecting it through the normal coordinated workflow.

Azure continues using the same image and endpoints, plus `/metrics` and JSON stdout logs.
Prometheus/Grafana/Mailpit here are a **local exercise stack**, not a claim that those services
have been deployed to Azure. To scrape the Azure app from a separately hosted Prometheus,
configure its public HTTPS host and `/metrics`; do not stop the public app to demonstrate alerts.
Do not mix local and production observations into one drift window. Keep one production
replica/process unless aggregation is redesigned.

## Stop and keep the evidence

```bash
docker compose logs api --no-log-prefix > results/monitoring-logs.txt
curl -s http://localhost:18000/metrics > results/monitoring-metrics.txt
docker compose -f compose.yml -f monitoring/compose.demo.yml down
```

`down` preserves the named volumes. The dashboard source is `monitoring/dashboard.json`, alert
provisioning is `monitoring/alerts.yml`; capture Mailpit firing/resolved messages and screenshots
for the presentation. Local datasets, registries, logs and `.env` stay ignored by Git and excluded
from the production image.
