# Submission review corrections

Reviewed 30 October 2026. Current submission claims differ from some historical experimental descriptions retained for development history.

- The team confirmed the name **Evan Tran & Toby Nguyen**, with both Evan Tran and Toby Nguyen as members. Earlier solo references omitted Evan and were corrected in the current write-up, README and demo script. Historical notes are retained as originally recorded.
- The team confirmed final public scores of **0.12944** for Part 1 and **0.16515** for Part 2. No final rank or private-test score is asserted. These team-confirmed results replace the earlier provisional score statements; an authenticated Kaggle record was not independently retrieved in this review.
- Canonical predictions match the selected `submission_v14.csv` and `p2_submission_v17_FINAL.csv`. Their hashes are in `final_submissions/manifest.json`. The earlier Part 1 repository copy differed row by row despite matching line totals; its original version remains in Git and a local backup. The user-selected source files were not changed.
- The target code subtracts recoveries, not a separate policy excess. The final model target is capped reported cost; individual claim-development weights were an earlier rejected approach.
- Target encodings were not fully nested inside outer validation folds. Blends, levels and variants were tuned using the same held-out prediction sets later used for reporting. Historical CV estimates may be optimistic.
- Public-leaderboard calibration uses adaptive feedback and assumptions about the unknown public subset. It is not independent validation or actual first-year market experience.
- Tweedie deviance targets the conditional mean. The previous theoretical distinction between a metric-optimal prediction and an unbiased expected cost was unsupported. Calibrated model predictions are not guaranteed unbiased.
- `src/pricing_tool.py` and the saved reference-policy CSV use an older v8 model. The saved loading ratio is technical/risk premium, not technical/written premium. The examples do not establish portfolio-wide unprofitability or final-v17 commercial performance.
- Current final-file checks passed. Generated-frame tests explicitly skipped absent intermediates. Training was not rerun. Missing components, incomplete final assembly and legacy paths are disclosed in `REPRODUCING.md`.

Older decision-log and experiment entries remain contemporaneous records. Current claims should be read from the revised write-up, validation reports and reproduction notes.
