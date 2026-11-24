# Local validation - 2026-11-24

The monitoring exercise was run against the production Dockerfile with an isolated synthetic
SQLite registry and the pinned Prometheus, Grafana and Mailpit images in Compose.

- Ruff and all 160 tests passed, including existing training, approval, rollback and API tests.
- `promtool check config` accepted the loaded configuration; the API target was UP.
- Grafana provisioned the six dashboard panels and all three rules with the required durations.
- The loaded training reference matched the approved model; normal traffic produced 13 feature
  drift values below 0.20. Synthetic shifted traffic raised the intended curves.
- Mailpit received firing AND resolved emails for data drift, high 5xx error rate and API down:
  six messages in total. No external SMTP account or actual email recipient was used.
- The local champion changed 1 -> 2 -> 1. Prometheus captured the model and prediction labels;
  startup and reference logs confirmed 1 -> 2 -> 1. The final API was ready on version 1.
- Five malformed replays returned 422 and produced correlated error logs without raw inputs.
- Grafana was inspected in a browser, including feature legends, the 0.20 threshold and model
  version. The local inbox was inspected for the firing and resolved messages.

Repeat with `python -m scripts.monitoring_check --alerts --rollback` after starting the demo
stack. Machine-readable evidence is written to the ignored `.exercise-results/check.json`;
logs and metrics were captured under the ignored `results/` directory.

This is local exercise evidence, not a claim that monitoring services are hosted on Azure.
Legacy DagsHub champions still need the documented hash-verified reference backfill before
real online drift is available. The public Azure app and its production alias were unchanged.
