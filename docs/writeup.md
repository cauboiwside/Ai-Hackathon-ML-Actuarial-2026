# Actuarial Pricing Across Markets with AI Assistance

**Team:** [TEAM NAME] — Toby (Khanh) Nguyen, [PARTNER NAME]
**Repository:** [GITHUB URL]  **Part 1 public LB:** 0.12944 (8th)  **Part 2 public LB:** [FINAL]

## 1. Approach

We priced three lines (private motor, home buildings, home contents) for a developed market (Part 1) and then adapted the approach to an emerging market with two years of history and a 2025 book that is 76% new business (Part 2). The workflow was the same in both parts: **data-quality scan → claims to ultimate → modelling → validation → level calibration → guardrails**, with AI (Claude) doing the exploration, coding and first-pass diagnostics, and the human reviewing, challenging and deciding. Every decision, who proposed it, and how it was verified is in `docs/decision_log.md` (about 40 entries).

**Target.** For each policy row, the capped cost of claims with a loss date in the row's period, net of excess and recoveries, excluding declined claims (caps: motor 250k, buildings 300k, contents 60k; the excess above the cap is spread back as a large-loss load of 1.044 for motor, ≈1.00 for home). The claims file is a 31 December 2024 snapshot, so recent years are immature. We first tried to bring each claim to ultimate individually (report-lag weights by peril, case development for bodily injury) and *reversed that decision* when a controlled experiment showed it hurt predictive accuracy (held-out D² 0.060 vs 0.064): per-claim weights of up to 5× add noise. The final approach models capped reported cost and applies IBNR, trend and development as aggregate level factors afterwards.

**Models (Part 1).** Three gradient-boosting models per line, blended: (i) a Tweedie (p = 1.5) model with a log-exposure offset, matching the scoring metric; (ii) a frequency × severity pair (Poisson count × Gamma size); (iii) a per-peril frequency × severity model (6 perils for buildings, 4 for contents, 5 for motor), the structure a home-pricing team would use. High-cardinality categoricals (vehicle model, occupation, region) use out-of-fold target encoding rather than native splits, which removed a large overfit on `make_model`. Validation is 4-fold cross-validation **grouped by policy**, so a policy's years never straddle train and test. Blend weights are fitted on out-of-fold predictions by Tweedie deviance; a partner model built independently is mixed in at 10%.

Held-out D² per line (Part 1): Tweedie 0.091 / 0.100 / 0.092; frequency-severity 0.094 / 0.102 / 0.095; per-peril 0.094 / 0.102 / 0.093; blend 0.095 / 0.105 / 0.097 (motor / buildings / contents).

## 2. Data quality, leakage and the things we chose not to use

The datasets contained planted problems. We caught them in two ways: a systematic scan, and diagnostics that produced impossible results.

- **Exposure in days.** About 900 training rows per line (Part 1) and 50 per line (Part 2) carried exposure as 365, 327, 89… instead of years. With a log-exposure offset, one such row carries 365× its weight: 0.4% of rows held 26% of predicted cost. We found it because actual-to-expected by exposure band was above 1 in every band yet below 1 overall, which is arithmetically impossible unless rows fall outside the bands. Fix: recompute exposure from the period dates. Lesson recorded: the scan now runs *before* modelling, and did in Part 2.
- **Scrambled codes.** `construction_type` appeared as Brick/BRICK/brick, STONE, NON_STANDARD and W/w/T/t for timber (Part 1) and as BRK/MIX/STN/WOD (Part 2); `occupation` in Part 2 carries a trailing space on every value. Normalised to the dictionary's four levels; W and T were merged only after checking they have identical claim frequency.
- **Target leakage.** `customer_service_contacts` is recorded *during* the policy period and is driven by claims: zero contacts means zero claims in 100% of rows, and in-sample D² jumps to 0.62. It is not knowable at quote time, so we excluded it. In Part 2 we verified the test column is scrubbed (its distribution matches training rows with no claims), so using it would also have been self-defeating.
- **Current premium.** `written_premium` is the insurer's own price; pricing off it is circular. Excluded on principle, and the experiment agreed: it reduces held-out D² (−0.006 motor).
- **Marketing fields** (campaign, segment, previous insurer): high model gain but flat claim rates across levels, i.e. noise. Excluded.
- **Gender (Part 2).** Present only in the emerging market. We treated it as a regulatory decision, not a modelling one: an insurer expanding from a market where gender rating is prohibited should not adopt it abroad. We measured the cost of excluding it: +0.0002 / −0.002 / +0.001 D² by line, i.e. nothing, because driver age band and vehicle already carry the signal.

## 3. Adapting to the new market

What changed: two years instead of four; 5% of the policy count; no credit band, exact ages (banded instead), mileage or telematics; a new region code system; 12% missing values; severities 20–40% lower; motor frequency higher (18% vs 14%); a far hailier climate (14–17 hail days vs 6); and a 2025 book that is 76% new business acquired from a broker versus 14% in the history.

What we carried over, and how. Four candidate models per line, all validated with policy-grouped cross-validation on the emerging rows only:

| Model | Idea | Motor | Buildings | Contents |
|---|---|---|---|---|
| Transfer (T) | developed-market model on the common feature set, re-levelled | 0.044 | 0.100 | 0.095 |
| Local (L) | emerging data only, heavily regularised | 0.044 | 0.048 | 0.072 |
| Offset (O) | local model trained on deviations from T (credibility weighting) | 0.044 | 0.107 | 0.098 |
| Pooled (P) | both markets together with a market flag and developed-market encodings as priors | 0.054 | 0.074 | 0.095 |
| **Blend** | weights fitted on held-out data | **0.058** | **0.107** | **0.104** |

The developed-market prior is worth twice the local data on the home lines; the credibility (offset) form is the best single model there. Motor is harder in this market because its strongest rating factors are missing, and the pooled model helps most. We tested autocalibration (Denuit, Charpentier & Trufin, 2021) and rejected it: with this little data the decile actual-to-expected ratios are noise.

**New business.** New-business policies cost 1.6× renewals in the local motor data (1.3× contents) versus 1.4× / 1.0× in the developed market, with very thin local exposure. We applied a credibility-weighted uplift, then measured the remaining level error on the public leaderboard (see §4) by segment: new business needed a further ×1.28, renewals ×1.13. We report this as the main uncertainty of the Part 2 price: the 2025 broker book is riskier than the two-year history, and only experience will settle by how much.

**Pricing tool.** The technical premium for the 20 reference policies is risk premium ÷ (1 − variable expense − commission) + fixed expense, using the `expenses` table (emerging: 85 CU fixed, 15% variable, 20% broker commission). The tool reports the *unbiased expected cost* separately from the metric-optimal prediction; the two differ because the Tweedie-1.5 score weights policies by μ^−½.

## 4. Use of AI, and how we verified it

AI wrote essentially all code and ran all experiments; the human directed, challenged and decided. The useful part of that arrangement was speed of *falsification*: around 60 model variants were tested in Part 1 and 20 in Part 2, each on the same held-out folds, and most were rejected. Examples of AI output that was wrong and caught: the per-claim ultimate weighting (reversed by experiment); a per-peril model whose frequency component started at one claim per policy-year because the base rate was missing from the offset (caught by a −6 held-out D²); a leaderboard "stretch" adjustment that validated offline and failed online (reverted). Examples where the AI's challenge was right: the leakage column, the exposure trap, and the finding that history features add nothing in this synthetic data.

**Leaderboard calibration without blind probing.** For a uniform multiplier *s* on the predictions, 1 − D²(s) = k + b·s^−½ + c·s^½, where k depends only on the truth and c is computable from the predictions. Three scored submissions identify k, b, c exactly; thereafter any new model needs one score to locate its optimal level, and single-segment perturbations identify per-line and per-segment levels (three equations, three lines). This replaced guesswork with a closed-form solve and was worth +0.004 in Part 1 and +0.008 in Part 2. It is disclosed here as what it is: the public 33% of the test set used as a stand-in for the first year of actual experience.

## 5. Guardrails, limitations and remaining uncertainty

**Automated tests** (`tests/`, 7 passing): submission format, every row exactly once, non-negative and finite, overall level within 25% of the training level, no row above 100× its line mean, exposure in years, categorical levels consistent between train and test.
**Governance:** a decision log with proposer and verification for every change; features excluded on principle regardless of score; a separate unbiased expected cost for pricing use; git history showing how the solution developed.
**Human review points:** large-loss cap levels; bodily-injury development; the new-business uplift; the gender decision; any leaderboard-derived level above ±20%.

**Limitations.** Bodily-injury development is a judgment (the data has no development history); the Part 1 experiment that applied it to the 2025 level lowered the score, so it is reported as a sensitivity (±25% on motor level) rather than applied. Motor in the emerging market is weakly predictable without mileage, exact age and telematics. Buildings swung from 512 to 308 per policy-year between the two local years; which year 2025 resembles cannot be known from the data. All leaderboard-derived levels rest on a 33% public sample. Finally, these are synthetic markets: past claims carry no information beyond the static risk factors here, which is not true of real books, and a real deployment would add experience rating.
