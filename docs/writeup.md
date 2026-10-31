# Actuarial Pricing Across Markets with AI Assistance

**Team:** Evan Tran & Toby Nguyen

**Authors:** Evan Tran and Toby Nguyen

**Code:** https://github.com/cauboiwside/Ai-Hackathon-ML-Actuarial-2026


## 1 Objective and modelling approach

We developed insurance claim-cost predictions for private motor, home buildings and home contents, first in a developed market and then in an emerging market with less history. The solution combines actuarial target construction, gradient-boosting models, cross-validation and calibration. The main adaptation was to retain developed-market risk patterns while allowing local claim levels and new-business experience to differ. AI assisted the exploration, implementation, diagnostics and drafting; we retained responsibility for the submitted solution.

### Data and target construction

Policies are joined to customer, address and weather data. Claims are matched by policy and loss date to the relevant policy period, duplicate claim IDs are removed, and repudiated claims contribute zero cost. The implemented cost is incurred-to-date less recoveries, floored at zero, with claim caps of 250,000 currency units for motor, 300,000 for buildings and 60,000 for contents. The code does not separately deduct policy excess. Exposure is recomputed from inclusive period dates and expressed in years.

An initial approach adjusted individual claims for reporting delay and development. The project experiments favoured modelling capped reported cost instead, because large individual adjustment weights added noise. Aggregate level adjustments then addressed trend, reporting development and large-loss loads. Bodily-injury development remains a judgement rather than an empirically identified factor, since the data do not contain a full development history.

### Part 1 model design

For each line we tested a Tweedie LightGBM model with variance power 1.5, a Poisson frequency model multiplied by a Gamma severity model, and a per-peril frequency-severity model. Exposure enters the expected-cost calculation through offsets or equivalent exposure-weighted rate modelling. Smoothed target encodings represent high-cardinality factors such as vehicle model and occupation. Out-of-fold predictions support model comparisons and blend selection.

The code uses four policy-grouped folds in Part 1, keeping repeated years of the same policy together. This is useful protection against repeated-policy overlap, but it does not make the evaluation fully independent. Target encodings were constructed once across folds, so encodings for an outer fold's training rows can depend on outer-validation labels. Blend weights and levels were also tuned on the same out-of-fold predictions used for reporting. Historical validation scores are therefore exploratory estimates and may be optimistic.


## 2 Adaptation to the emerging market

The emerging market provides two years of history and a smaller training sample. Several developed-market features are absent or coarser, including exact ages, mileage, telematics and credit information. Category codes differ and missingness is greater. The project diagnostics also identify a substantial shift toward new business in the prediction book. These changes weaken a model trained only on the small local history and make portfolio composition a central calibration uncertainty.

### Transfer and local evidence

We harmonised shared features, converted developed-market ages to the available bands, normalised construction codes and occupation text, and omitted unavailable factors. Developed-market region identifiers were not treated as interchangeable with emerging-market regions. Five policy-grouped folds on emerging-market rows compare transfer, local, offset and pooled approaches.

| Candidate | How it uses the two markets |
| --- | --- |
| Transfer | Train on developed-market common features and adjust the local level. |
| Local | Train on emerging-market experience with stronger regularisation. |
| Offset | Fit local deviations around a transferred prediction used as a prior. |
| Pooled | Combine both markets, with a market indicator and greater local weight. |

Frequency-severity and per-peril versions of the transferred prior expanded the ensemble. A regularised Tweedie GLM provided an interpretable benchmark and entered the contents blend. The saved pre-segment ensemble diagnostics report D-squared of approximately 0.063 for motor, 0.112 for buildings and 0.111 for contents. These are recorded training diagnostics, with the validation limitations described above; they are not private-test or independent deployment results.

The experiments favoured transfer-informed models on the property lines. Separating frequency and severity was a useful way to allow claim levels to differ while retaining shared risk patterns. Local new-business and regional adjustments were treated cautiously because sparse experience makes them unstable. One multi-factor balance correction improved its fitting sample but deteriorated under a further cross-fit check and was rejected.

### Reported competition results

Our final public-leaderboard scores, confirmed by the team, are 0.12944 for Part 1 and 0.16515 for Part 2. The project log also records the Part 2 v17 score and an initial Part 2 score of 0.1379. Public scores are reported separately from the historical cross-validation diagnostics because the evaluation samples and calibration procedures differ. No private-leaderboard result or final rank is claimed for either part.


## 3 Major decisions and AI assisted workflow

### Data quality and feature guardrails

The data-quality work corrected exposure values recorded in days rather than years and inconsistent construction-type labels. An exposure error is especially damaging because the exposure offset scales the entire prediction. The workflow was changed so that these checks run before modelling. Part 2 also required whitespace removal and explicit alignment of category definitions across markets.

Customer-service contacts were excluded because they can reflect claims occurring during the insured period and would not be available at quotation. Written premium was excluded from the primary risk model to avoid learning the insurer's existing price. Marketing and administrative fields were also omitted. Gender was excluded in the emerging-market pipeline as a governance choice; this write-up does not assert a legal requirement for an unspecified jurisdiction. Weather and household features would require an explicit quote-time availability check before live use.

### How AI contributed and how decisions were checked

The decision log attributes substantial exploration, coding and diagnostics to Claude, with human direction and review. AI accelerated candidate generation and helped investigate unusual results. It did not remove the need to check the target definition, feature availability, offsets, validation design or final files. Git history and the decision log retain the development sequence, including rejected approaches.

Examples of corrections include reversing individual claim-development weights after an adverse experiment, fixing a missing frequency base-rate offset after implausible validation results, and rejecting calibration changes that did not hold up under further checks. A final Codex-assisted review checked the supplied CSVs and revised this write-up, including correcting an unsupported distinction between a Tweedie-optimal target and an unbiased expected-cost target. Tweedie deviance targets the conditional mean; a calibrated fitted prediction is not proof of unbiasedness.

### Use of public leaderboard feedback

The final solution uses public scores to calibrate overall, line, new-business and selected rating-segment levels. The calibration scripts exploit the analytical form of Tweedie deviance under prediction multipliers. The curve is exact for a fixed set of scored rows, but practical estimation uses full-test prediction summaries as a proxy for the unknown public subset and receives rounded scores. Its conclusions therefore depend on representativeness and numerical assumptions.

Repeated public-score feedback makes the public leaderboard an adaptive tuning sample. It is not independent validation and is not equivalent to observing a new year of actual claims. The final Part 2 segment adjustments are disclosed as public-sample calibration. Their apparent improvement may not persist on the private set or in a different portfolio; the public score should not be interpreted as an unbiased generalisation estimate.


## 4 Final artifacts guardrails and limitations

### Verification of the selected prediction files

The final-file review on 30 October 2026 compared both selected CSVs with the supplied official sample submissions and test-policy files. Each has exactly the required columns, every required ID once, matching sample order, and finite positive predictions. No hidden test-claim outcomes were available, so this establishes submission integrity rather than predictive accuracy.

| Check | Part 1 v14 | Part 2 v17 |
| --- | --- | --- |
| Prediction rows | 626,639 | 209,063 |
| Schema | row_id, prediction | row_id, prediction |
| Missing or duplicate IDs | 0 | 0 |
| Missing, non-finite or negative predictions | 0 | 0 |
| Official sample ID set and order | Match | Match |

The code package uses final_submissions/part1_prediction_file.csv and final_submissions/part2_prediction_file.csv as the canonical artifacts. SHA-256 hashes tie these copies to the two selected source files. An earlier Part 1 repository copy had different row-level predictions despite identical aggregate totals; it was preserved separately before the selected v14 became canonical. Portable validation commands and the detailed audit are included with the code.

### Pricing illustration and operational guardrails

The pricing example converts a risk premium to a technical premium by adding catastrophe and fixed-expense allowances and dividing by the fraction remaining after variable expenses, commission, tax and target margin. Its saved reference-policy output uses an earlier v8 model. The reported loading ratio is technical premium divided by risk premium, not divided by written premium. These examples illustrate expense effects; they do not establish that the entire broker book is unprofitable or demonstrate final-v17 commercial performance.

Before deployment we would require nested encoding and blend evaluation, an untouched temporal or external holdout, quote-time feature checks, sensitivity analysis for development and large losses, and review of large calibration movements. Positive-output checks protect the file format but cannot establish fair, adequate or stable insurance rates.

### Reproducibility and remaining uncertainty

The repository preserves the original development history and supplies code, decision records, dependency guidance, final CSVs and validation tooling. Competition data are kept outside the repository. The exact supplied Part 1 v14 construction recipe and the complete final Part 2 segment-correction sequence are not fully captured as executable pipelines; several legacy scripts also retain environment-specific paths. The current package supports exact final-artifact verification, while end-to-end retraining remains incomplete. These are reproducibility limitations of the submitted solution.

Sparse emerging-market experience, missing motor predictors, uncertain claims development, changing new-business mix and public-sample calibration remain the principal modelling risks. Results on these synthetic markets do not establish how the method would perform on a live insurance portfolio. The next priority would be independent validation and completion of the recorded training and final-assembly recipes before making production decisions.
