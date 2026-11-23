# Experiments

All runs are tracked on the shared DagsHub MLflow server
(`https://dagshub.com/zorobabel33/atp-match-predictor.mlflow`). Results below were produced on
2026-11-22 with a clean git tree: model selection and real-data runs from commit `656322e`,
controlled staging runs from commit `f1dae74`.

## Primary metric

ROC-AUC on the held-out evaluation season, **higher is better**.

The web app shows the probability that player A wins. ROC-AUC measures whether the model ranks
winners above losers across all thresholds, so it rewards useful probabilities rather than one
cut-off, and it does not depend on which player is called A (the label is balanced by the random
player order). Log loss (quality of the probabilities themselves) and accuracy (share of correct
predictions at 0.5, easy to explain to a user) are logged next to it.

Every run also logs `reference_rank_roc_auc`: the ROC-AUC of the rule "the better ranked player
wins" on the same rows. A model is only useful if it beats this rule.

## Quality gates

Fixed in `ml/policy.py` before any model was evaluated on 2022 or 2023. They were calibrated on
training seasons only: fit on 2016-2020, validate on 2021 (both inside v1 `train.csv`).

| Gate | Value | Evidence |
|---|---|---|
| Quality floor | ROC-AUC >= 0.68 | The ranking rule reaches 0.673 on 2021: a model must do better. |
| Minimum improvement over the champion, on the same rows | +0.001 | Bootstrap standard deviation of a paired ROC-AUC difference on 2021: 0.0005. The gate is about twice that noise. Adding one season (2016-2019 vs 2016-2020 training) gained +0.0016. |
| Retraining trigger | new validated dataset version, champion below 0.68, or a drop >= 0.02 from its approval ROC-AUC | A new labeled season is the expected trigger; the other two catch degradation. |

The first release (bootstrap) only has to pass the floor. A candidate that fails a gate is not
registered, the champion is kept and the reason is logged (`reason` tag and `decision.json`).

## Model selection

Experiment `atp-model-selection`, `python -m ml.compare_models`. Fit 2016-2020, validate 2021.

| Run | Model | ROC-AUC | Log loss | Accuracy | run_id |
|---|---|---|---|---|---|
| rank-reference | better ranked player wins | 0.6726 | - | - | `6ac102bf058345d98a647f90875815bc` |
| logreg-C0.1 | LogisticRegression, C=0.1 | 0.7108 | 0.6234 | 0.6497 | `0fc91e8727e3459bac87c55b855f385c` |
| logreg-C1 | LogisticRegression, C=1 | 0.7108 | 0.6235 | 0.6505 | `d0ae7cb86c594b5183b515250c5b8c6d` |
| logreg-C10 | LogisticRegression, C=10 | 0.7108 | 0.6235 | 0.6505 | `2db291af78e74817bf87b3811392e078` |
| hist-gradient-boosting | HistGradientBoostingClassifier | 0.6985 | 0.6281 | 0.6334 | `7c07823b988d4f42b2569e721b51f4af` |

Choice: logistic regression with one-hot categories and standardized numeric features, `C=1.0`.
It has the best ROC-AUC, is insensitive to `C`, is fast to retrain in CI, and its coefficients
are easy to explain. Gradient boosting does not help with these few pre-match features.

## Pipeline runs on the real data

Experiment and model `atp-match-winner-dev` (local runs that validate the pipeline without touching
the production model `atp-match-winner`). Data pulled from the Azure DVC remote.

| Run | Data | Decision | Champion | Candidate | Ranking rule | run_id |
|---|---|---|---|---|---|---|
| v1 | v1, eval 2022 | `bootstrapped`, version 1 | - | 0.7002 | 0.6783 | `7d6d8eb54f454442a8d19be5c6664f89` |
| v1 again | v1, eval 2022 | `retain_no_retraining` | 0.7002 | - | 0.6783 | `98a373491fe14548a1a37134bdcc441e` |
| v2 | v2, eval 2023 | `retain_rejected` | 0.7000 | 0.6994 | 0.6640 | `c1b84e2668c04ef4b8de22967ac9dadc` |

v1 champion on 2022: ROC-AUC 0.7002, log loss 0.6302, accuracy 0.643.

v2 retrained because a new dataset version arrived (`retraining_trigger=new_dataset_version`). On
the same 2023 rows the candidate trained on 2016-2022 scored 0.6994 and the v1 champion 0.7000: the
improvement, -0.0006, is below +0.001, so the champion is retained. One more season does not add
information to a model limited to ranking, points, age, height and hand. The gates were not changed
after this result.

**Production model choice:** the v1 logistic regression, approved at ROC-AUC 0.7002 on 2022 and
still 0.7000 on 2023, about 0.036 above the ranking rule on unseen seasons.

## Controlled promotion and rollback

The real v2 does not produce a promotion, so promotion and rollback are demonstrated on a separate
model name, `atp-match-winner-staging`, with synthetic releases from `tests/synthetic.py`
(`python -m tests.synthetic <dir>`, then `DATA_DIR=<dir>`). In v1 the winner depends on age, in v2
on ranking: a concept drift that the v1 champion cannot follow. The pipeline, gates and registry are
the same; runs are tagged `data_source=local` and log no DVC hash.

| Run | Decision | Champion | Candidate | run_id |
|---|---|---|---|---|
| v1 | `bootstrapped`, version 1 | - | 0.9357 | `7320ab2ea78a4a1695b3add915af1ed1` |
| v2 | `promoted`, version 1 -> 2 | 0.5723 | 0.9572 | `396bf67b30c64a1097e74f5591237d82` |
| rollback 1 | `rolled_back`, version 2 -> 1 | - | - | `a7830dc252a047c796cb8fbf6f49dee5` |
| rollback 7 | `failed`, version 7 does not exist, champion unchanged | - | - | `c8e7adf550c84be1952f372f28847785` |

Both versions of `atp-match-winner-staging` are approved, so a live rollback can switch between 1
and 2.

The same decisions (bootstrap, retain, promote, reject, invalid data, older window, rollback,
refused rollback) are covered by `tests/test_pipeline.py` on a temporary local registry in CI.

## Reproduce

```bash
python -m ml.compare_models
MODEL_NAME=atp-match-winner-dev MLFLOW_EXPERIMENT_NAME=atp-match-winner-dev \
  python -m ml.pipeline --dataset-version v1
```

Production runs use the CT workflow (`.github/workflows/ct.yml`) with the model `atp-match-winner`.
