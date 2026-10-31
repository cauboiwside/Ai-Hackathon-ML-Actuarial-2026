# Final demo recording script

Team: Evan Tran & Toby Nguyen. Target: 4 minutes 30 seconds to 4 minutes 45 seconds at a measured speaking pace. Rehearse once with a timer; only the first five minutes are graded. Record a screen demonstration with your own narration. This is a script, not a recorded video.

## 0 minutes to 40 seconds

Show the README and final prediction files.

Hello, we are Evan Tran and Toby Nguyen. Our project predicts claim costs for motor, home buildings and home contents. In week one we worked with a developed market. In week two we adapted the approach to an emerging market with less data and only two years of history. We will explain the models, the market adaptation, the AI workflow, our results and the limitations.

Our final public scores are 0.12944 for Part 1 and 0.16515 for Part 2. The selected prediction files have also been checked against the official sample IDs and row order.

## 40 seconds to 1 minute 25 seconds

Show the target and model paragraphs in the write-up, then `src/train_v2.py`.

We joined policies to customer, address and weather information, matched claims to policy periods, and recomputed exposure in years. The final target uses capped reported claims net of recoveries. An earlier approach used large individual development weights, but experiments favoured a simpler reported-cost target with aggregate level adjustments.

For each line we tested a Tweedie gradient-boosting model, a frequency-times-severity model, and a per-peril version. Held-out predictions supported model comparisons and blending.

## 1 minute 25 seconds to 2 minutes 10 seconds

Show the decision log and feature exclusions in `src/common.py`.

AI helped write code, explore data and investigate diagnostics. The safeguard was checking suggestions against experiments. The workflow caught exposure values recorded in days, inconsistent category codes and a frequency-model offset error. Some promising changes were rejected after further checks.

Customer-service contacts were excluded because they can occur after a claim and would not be available at quotation. Written premium was excluded from the primary risk model, and gender was omitted as a governance choice. The decision log and Git history preserve these choices. We remain responsible for the submitted output.

## 2 minutes 10 seconds to 3 minutes

Show the four model approaches on page two of the write-up and `src/p2_train.py`.

The emerging market had fewer observations, missing predictors, coarser age bands and a different category system. Its prediction book also had a much larger share of new business.

We harmonised features and compared four approaches: transfer developed-market models and adjust their levels; train locally; fit local deviations around transferred predictions; or pool both markets with a market indicator. Frequency-severity and per-peril transfer models expanded the ensemble. Transfer-informed models were especially useful on property lines. Motor remained harder without its richer predictors.

## 3 minutes to 3 minutes 45 seconds

Show the reported results and public calibration paragraphs in the write-up.

The project log records an initial Part 2 score of 0.1379 and a final v17 score of 0.16515. We also used public-leaderboard feedback to calibrate line and rating-segment levels. That makes the public leaderboard a tuning sample, not independent validation. Its gain may not carry over to the private set.

Historical target encodings were not fully nested inside each validation split, and blending was evaluated on the data used to tune it. We disclose that limitation because cross-validation scores may be optimistic. Our public scores and historical cross-validation diagnostics should be interpreted separately.

## 3 minutes 45 seconds to 4 minutes 35 seconds

Run `python3 scripts/validate_submissions.py`, show the passing result, then show reproduction notes.

The file checks verify columns, unique IDs, row counts and finite nonnegative predictions. Comparisons with the official sample files also passed for both parts. These checks establish file integrity, not predictive accuracy.

The repository includes code, decisions, final files, hashes and validation commands. Full retraining remains incomplete because some intermediate files and final assembly steps were not captured. Before deployment we would complete those recipes, rebuild encodings inside every split and evaluate on an untouched time period. Sparse local experience, claims development and portfolio shift remain the main risks. Thank you.
