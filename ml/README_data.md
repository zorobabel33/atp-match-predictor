# Data

## Source and license

- Data: ATP tour-level match results compiled by Jeff Sackmann, originally published at
  `https://github.com/JeffSackmann/tennis_atp`.
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribution to Jeff
  Sackmann, non-commercial use only, derived datasets under the same license. This project is an
  academic, non-commercial use, and the prepared datasets below keep the same license.
- Snapshot: the original repository was no longer public when this project started (November 2026).
  We use the archival mirror
  [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive), which
  redistributes the same data under the same license (upstream ATP snapshot from June 2026). It is
  pinned to commit `83733587353df8a41f2fd4f516147d5aa83f5a8d` in `ml/prepare_data.py`.
- Raw files used, `atp/atp_matches_<season>.csv` (SHA-256):

| Season | SHA-256 |
|---|---|
| 2016 | `86dda3871cb79d3226a047774c8fc97b86267cfe96709dea832a347dfbf81684` |
| 2017 | `4bb295788c37487b120cd39171d413cef0ebc6f4c2d9fa40b7261f5143f3719f` |
| 2018 | `0363dc6012190f337fcf34b4ca443634f192c4670cc6c14c94342a54b9adf7eb` |
| 2019 | `7f1a21451ff622bb0c6a09690898db46d51e59dcdf08086d369753822e1d5478` |
| 2020 | `985a917b72d5efb7dd5ef9df8ecc58c637ed6b88561ccad364653af71ac12da0` |
| 2021 | `b9b1d31a4b0b9273b8f338cbb1347c5a847ad2361334ec760a286f5990fba347` |
| 2022 | `7190042565ab8400e74f44fda60700acb36ddf3c82d393f331a07589e05b5f52` |
| 2023 | `9b9671aa7c8156e74c2e4466675b7ed4e762bb8a58b7237a30daefd84ffceff5` |

## Preparation

`python -m ml.prepare_data --version v1` (or `v2`) downloads the raw seasons into `raw/` (not
tracked) and writes `data/<version>/train.csv` and `data/<version>/evaluation.csv`.

- Pre-match information only: surface, tournament level, best-of, and for each player the ranking,
  ranking points, age, height and hand. Match statistics (`minutes`, `w_*`, `l_*`) are dropped.
  `score` is only read to detect walkovers.
- Player order: `winner_*` / `loser_*` would reveal the label, so each match is turned into
  `player_a_*` / `player_b_*`. Player A is the winner when the first byte of
  `sha256("42-<match_id>")` is even. This is a coin flip with a fixed seed that depends only on the
  match, so a match keeps the same order in every release. Label `a_wins`: 1 if player A won.
- Rows removed, in this order (counts for the 2016-2023 seasons used by v2):

| Rule | Rows |
|---|---|
| Walkover (`W/O`): the match was not played | 132 |
| Any missing feature (mostly Davis Cup, unranked players, unknown height) | 513 |
| Category outside the contract: Davis Cup (level `D`), `Carpet`, hand `U` | 1,144 |
| Numeric value out of range (see `ml/contract.py`) | 0 |

  Retirements (`RET`) are kept: the official result exists and a retirement is not known before the
  match.
- Allowed values and ranges are defined once in `ml/contract.py`. The training pipeline validates the
  same contract before fitting.

## Split design

Splits follow time: a model is always evaluated on a later season than any match it was trained on.
The split uses the season file, and the preparation checks that the last training `tourney_date` is
before the first evaluation `tourney_date`.

| Version | Training | Evaluation | Train rows | Evaluation rows | `a_wins` rate (train / eval) |
|---|---|---|---|---|---|
| v1 | 2016-2021 | 2022 | 14,441 | 2,689 | 0.508 / 0.489 |
| v2 | 2016-2022 | 2023 | 17,130 | 2,734 | 0.505 / 0.501 |

Last training date / first evaluation date: v1 `20211115` / `20220103`, v2 `20221114` / `20230102`.

## What changed between v1 and v2

v2 simulates the arrival of a new season of labeled matches. The 2,689 matches of 2022, which were
v1's held-out evaluation set, become additional training data, and 2023 becomes the new held-out
set. The first 14,441 rows of v2 `train.csv` are identical to v1 `train.csv`. No model of either
release is fitted or tuned on 2023 matches.

The candidate trained on v2 and the v1 champion are both scored on the same v2 `evaluation.csv`
(2023).

## Versions

| Version | Git tag | Commit | DVC pointer | DVC hash (directory) |
|---|---|---|---|---|
| v1 | `data-v1` | `cddad11432d838f8e4b26ab3a8886975782edfc6` | `data/v1.dvc` | `489cf3860f33ee4343e88d6501934d80.dir` |
| v2 | `data-v2` | `cf5a6528654f304eaf050833604b2e73275570eb` | `data/v2.dvc` | `e36a8fed661606ca81f4aa9ce9324231.dir` |

File hashes (MD5, as stored by DVC):

| File | MD5 |
|---|---|
| `v1/train.csv` | `43b9c0a21595b1bb8509e302f8af6906` |
| `v1/evaluation.csv` | `05d44cb001d1b7f2f7a36fb50d6f60b9` |
| `v2/train.csv` | `c6bb8e6d8f264a17d13221f10d8e4a15` |
| `v2/evaluation.csv` | `439b8a270abf573f898f7f51f18e55d5` |

## Storage

The DVC remote is the Azure Blob Storage container provided by the course: storage account
`wut2026`, container `dvc`, prefix `atp-match-predictor/` (`azure://dvc/atp-match-predictor`). Access
uses the course SAS token, which is never committed:

```bash
dvc remote modify --local azure sas_token "$AZURE_SAS_TOKEN"   # written to .dvc/config.local
git checkout data-v2   # or use the current branch, both pointers are present
dvc pull data/v2.dvc
```

In GitHub Actions the token comes from the `AZURE_SAS_TOKEN` secret. The course token expires on
2026-12-31.
