# Reproduction status and file validation

The final CSVs can be checked independently of model training. The archived
source records modelling experiments and decisions, but this checkout is not a
complete, one-command reproduction of the final fitted models. No model was
retrained during the submission-file review.

## Check the delivered prediction files

From the repository root, run:

```sh
python3 scripts/validate_submissions.py --report /tmp/iaa-validation.json
python3 -m unittest discover -s tests -v
```

These commands check the canonical files in `final_submissions/`:

| File | Expected rows |
| --- | ---: |
| `part1_prediction_file.csv` | 626,639 |
| `part2_prediction_file.csv` | 209,063 |

The validator checks exact columns, nonempty unique IDs, finite nonnegative
predictions and expected row counts. It records SHA-256 hashes, counts and
prediction summaries by line. Without an official sample, it explicitly reports
that exact ID coverage and order have not been verified. A valid CSV does not
establish predictive accuracy, a leaderboard score or compliance with competition
rules.

For exact ID checks against the official files, supply the corresponding samples:

```sh
python3 scripts/validate_submissions.py \
  --part1-sample /path/to/public/sample_submission.parquet \
  --part2-sample /path/to/emerging/sample_submission.parquet \
  --report /tmp/iaa-validation.json
```

CSV samples are also supported. CSV-only checks need only the Python standard
library. Parquet samples require `pyarrow`. The report gives missing/extra ID
counts and whether row order matches; a different order with identical IDs is
reported as a warning. The tool never rewrites predictions. `--part1` and
`--part2` allow explicit alternative CSV paths, avoiding automatic selection by
filename or version number.

The submission and validator tests run with standard-library `unittest` or with
`pytest`. Generated-frame checks are explicitly skipped if the six Part 1 model
frames are absent. A skip means the training-data checks did not run, and must
not be presented as evidence that training inputs passed QA.

During this review, both canonical CSVs passed the validator against their
official parquet samples: zero missing or extra IDs, matching row order, and
no schema or prediction-value errors. The portable test suite passed nine tests;
the generated-frame test class was skipped because the frames were absent.
These checks validate the files delivered here and do not validate hidden-target
performance or reproduce the fitted models.

## Dependencies

For the historical modelling scripts and parquet checks, create an isolated
environment and install the direct dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` lists dependency ranges inferred from imports and API use. It
is not a lockfile recovered from the original model run. The current review ran
file validation and tests, not a complete training-environment compatibility
check.

## Historical pipeline and outstanding gaps

Keep organiser-provided datasets outside the Git repository and obtain them
through the competition's permitted distribution channel.

| Stage | Source entry points | Required inputs or outputs |
| --- | --- | --- |
| Part 1 preparation | `src/prep.py`, then `src/level.py`, then `src/features.py` | Developed-market parquet data; writes `work/` claim and model frames |
| Part 1 models | `src/train_v2.py`, `src/train_fs.py`, `src/train_fs_peril.py` | Generated model and claim frames; writes component predictions |
| Part 1 assembly | `src/build_final.py`, `src/blend.py`, `src/blend2.py` | Blend weights, matched component files and, for historical partner blends, external partner predictions |
| Part 2 preparation | `src/p2_prep.py` | Developed and emerging raw data; writes `work_p2/` model frames |
| Part 2 models | `src/p2_train.py`, `src/p2_pooled.py`, other `src/p2_*.py` experiments | Generated frames; creates/updates OOF and test component columns |
| Part 2 assembly | `src/p2_build.py`, `src/p2_build_v3.py` | OOF/test component parquet files, version-specific stack JSON, segment multipliers and initial prediction CSV |

This table is a source map, not a verified command sequence that recreates the
final files. Important gaps remain:

- The reviewed checkout has no `work/` generated model frames or fitted models,
  and no usable `work_p2/` intermediates. Do not infer their existence from
  historical references or Git history.
- Several historical Part 1 blends depend on external partner CSVs or matched
  component parquet files that are not included. Some assembly scripts require
  these files even when a corresponding blend weight is zero.
- The source does not provide a complete executable recipe for every later
  version through the supplied Part 1 v14 and Part 2 v17 final files. Exact
  version-specific component files, stacking weights and segment adjustments
  must be recovered before claiming byte-for-byte regeneration.
- Later calibrations use public leaderboard feedback. Recreating them requires
  the recorded probe scores and submitted prediction versions; local grouped
  validation alone does not reproduce that feedback. Such calibration can
  overfit the public sample, and its gain is not a private-test guarantee.
- Most historical scripts still use `/home/claude/data_raw/public/`,
  `/home/claude/data_p2/emerging/`, `/home/claude/iaa-pricing/work/` or
  `/home/claude/iaa-pricing/work_p2/` directly. Some also reference
  `/home/claude/partner/` and `/mnt/user-data/outputs/`. Port those paths
  deliberately before running the relevant script. They are not all redirected
  by the setting below.

`src/common.py` now defaults its shared `W` setting to the repository's `work/`
directory and permits an override:

```sh
export IAA_WORK_DIR=/absolute/path/to/generated-part1-work
```

The generated-frame tests use the same override. This affects only scripts that
import `W` from `common`; it does not configure raw-data locations, independent
`W` assignments or Part 2 `W2` paths. Path configuration was deliberately limited
to avoid changing the historical models during a file-integrity review.

## Version-control provenance

Preserve existing commits and genuine experiment history. A packaging or
documentation commit should describe the changes actually made; it cannot
retroactively establish missing training provenance. Final CSV hashes in the
validation report identify exactly which bytes were reviewed.
