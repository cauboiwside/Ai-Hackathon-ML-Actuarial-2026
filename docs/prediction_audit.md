# Prediction file audit

Read-only audit of the two user-selected CSVs and their existing repository copies against locally supplied sample submissions and test files. Predictions were not modified or retrained.

## Results

| Check | Part 1 | Part 2 |
| --- | ---: | ---: |
| Selected file | submission_v14.csv | p2_submission_v17_FINAL.csv |
| Rows | 626,639 | 209,063 |
| Unique IDs | 626,639 | 209,063 |
| Columns | row_id, prediction | row_id, prediction |
| Missing values | 0 | 0 |
| Duplicate IDs | 0 | 0 |
| Nonnumeric / infinite / negative predictions | 0 / 0 / 0 | 0 / 0 / 0 |
| Zero predictions | 0 | 0 |
| Missing / extra IDs vs local sample | 0 / 0 | 0 / 0 |
| Missing / extra IDs vs local test files | 0 / 0 | 0 / 0 |
| Matches local sample ID order | True | True |
| Repository copy byte identical | True | True |
| All predictions below 100× own line mean | True | True |
| Structural result | PASS | PASS |

The structural checks pass for the exact selected files. They establish format and coverage, and do not establish predictive accuracy. Repository consistency is reported separately.

## Repository consistency

- part1: selected file and existing repository copy are byte identical.
- part2: selected file and existing repository copy are byte identical.

The originally supplied Part 1 repository copy had the same 626,639 IDs and line totals as the selected v14 file, but all 626,639 predictions differed (mean absolute difference 6.489751, maximum 1638.784509, correlation 0.9993501). It was preserved at `/Users/nghiatranyeuem/Downloads/iaa_submission_review/originals/repository_part1_before_review.csv` (SHA-256 `becd128f8d3a46abf84d3110c30e098bf9f14aae732ff3ad0c750eef6e6e3691`). The canonical repository path now contains the explicit user-selected file byte for byte. The prediction values in the selected files were not changed. Equal aggregate totals do not make individual predictions equivalent, and these diagnostics do not establish which file has a higher hidden-test score.

## Prediction distributions

| Part | Line | Rows | Mean prediction | Median | 99th percentile | Maximum | Predicted cost / exposure year |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| part1 | motor | 254,986 | 455.683 | 384.594 | 1,855.582 | 7,527.823 | 581.012 |
| part1 | buildings | 184,155 | 438.998 | 347.881 | 2,387.438 | 21,711.295 | 538.083 |
| part1 | contents | 187,498 | 112.004 | 89.555 | 561.163 | 3,082.277 | 139.888 |
| part2 | motor | 90,521 | 604.087 | 514.338 | 1,911.652 | 3,547.687 | 692.123 |
| part2 | buildings | 52,295 | 371.653 | 265.457 | 2,286.733 | 7,514.306 | 410.238 |
| part2 | contents | 66,247 | 94.268 | 73.502 | 493.560 | 2,473.218 | 105.818 |

Predictions are expected policy-period loss cost. The last column divides aggregate predictions by exposure recomputed as `(period_end - period_start + 1 day) / 365`, consistent with the preparation scripts. It is not an observed loss ratio or a score.

## Data quality context

- part1 motor: 254,986 test rows, 15.25% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.
- part1 buildings: 184,155 test rows, 15.15% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.
- part1 contents: 187,498 test rows, 15.16% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.
- part2 motor: 90,521 test rows, 76.03% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.
- part2 buildings: 52,295 test rows, 75.95% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.
- part2 contents: 66,247 test rows, 76.14% new business; 0 raw exposure values above 1.005, 0 nonpositive exposures, 0 missing exposures. Normalized exposure range: 0.002740–1.000000 years; largest raw-vs-date difference 0.00004932 years.

The small raw-vs-date differences are consistent with rounding: no test rows differ by more than 0.0001 years. These diagnostics do not show a test exposure error. The source scripts explicitly rebuild exposure from dates; the submitted CSV contains only IDs and predictions, so the exact upstream run cannot be reconstructed from it.

## Test suite and reproducibility

The initially supplied suite was not run: the selected Python environment had no pytest installed (No module named pytest). These observations describe the original project before portability changes by the root reviewer.

- tests/test_submission.py and tests/test_data_quality.py hard-code /home/claude paths; these paths and work artifacts are unavailable in the supplied project.
- test_submission.py selects a lexicographically sorted submission_v*.csv rather than the two user-selected final paths; it does not validate Part 2.
- Exposure and category consistency tests require model-frame artifacts, which are absent from the supplied project.
- The exact two selected CSVs have instead been independently checked here for schema, IDs, finite nonnegative predictions, outlier guardrails, and byte identity with repository copies.

## Performance evidence and limitations

Hidden test target values and authenticated Kaggle submission records are unavailable. This audit cannot recalculate D2, establish a leaderboard score/rank, or prove either file was submitted to Kaggle.

The README reports Part 1 public D2 0.1294 (8th of 37, team entry) and Part 2 public D2 0.1618. These are documentary claims, not independently verified scores for the selected CSVs. Earlier metrics in `docs/cv_results_v1.json` refer to an earlier model version.

- Local sample/test files were treated as supplied competition references; their upstream authenticity was not independently verified online.
- Finite nonnegative costs and complete IDs establish structural submission validity, not predictive accuracy or actuarial adequacy.
- Summary costs per exposure year use exposure recomputed from inclusive policy dates, matching src/features.py and src/p2_prep.py; no target truth is used.
- Existing code includes public-leaderboard-driven global and segment calibration (lb_calib.py, lb_lines.py, p2_lb_solve.py). Such calibration can overfit the public leaderboard and should be disclosed accurately.
- The original build scripts depend on missing intermediate work/ and work_p2/ artifacts, hard-coded Linux paths, model weights and component predictions. The final CSVs alone cannot prove end-to-end reproducibility.
- The README points to different historical work/ final filenames. The two final_submissions files can be identified unambiguously by the hashes recorded in this audit.

## File provenance

### part1

- Selected CSV: `/Users/nghiatranyeuem/Downloads/submission_v14.csv`
- SHA-256: `6047189045f6746a79f3b22b1be757741245261566bd0bf17363fd42047878b5`
- Repository copy: `/Users/nghiatranyeuem/Downloads/iaa-pricing/final_submissions/part1_prediction_file.csv`
- Repository copy SHA-256: `6047189045f6746a79f3b22b1be757741245261566bd0bf17363fd42047878b5`
- Local sample: `/Users/nghiatranyeuem/Downloads/public/sample_submission.parquet`
- Local sample SHA-256: `9529f3b6a679d3410350baa7c2ef3fb085d1d8843a838360c5852dfe820d4d95`

### part2

- Selected CSV: `/Users/nghiatranyeuem/Downloads/p2_submission_v17_FINAL.csv`
- SHA-256: `8e068da53ff159aaba2182383b7212b6b78bbd834b9b613cc1f80f271b9bee9b`
- Repository copy: `/Users/nghiatranyeuem/Downloads/iaa-pricing/final_submissions/part2_prediction_file.csv`
- Repository copy SHA-256: `8e068da53ff159aaba2182383b7212b6b78bbd834b9b613cc1f80f271b9bee9b`
- Local sample: `/Users/nghiatranyeuem/Downloads/iaa-ai-assisted-hackathon-2026-part-2/emerging/sample_submission.parquet`
- Local sample SHA-256: `160fc59043f1012337605c8d8446c0efbf81458e046cf02df9186d6f909d59ef`

All detailed diagnostics and boolean checks are recorded in `prediction_audit.json`. No source data rows are reproduced in this report.
