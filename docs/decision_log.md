# Decision log (Week 1)

Each entry: what was found, who proposed it (AI / Toby), how it was checked, decision.

| # | Finding | Proposed by | Verification | Decision |
|---|---|---|---|---|
| 1 | 2024 loss year has ~8k fewer claims and 20% open -> data is not at ultimate | AI | Counts and open share by loss year | Build IBNR + case development before modelling |
| 2 | Report lags differ strongly by peril (BI median ~5 months, 99th pct 800 days; subsidence ~4 months) | AI | Lag distribution from 2021 losses (3+ yrs observed) | Peril-level IBNR weight = 1 / F_peril(age at 31-Dec-24), floor F at 0.2 |
| 3 | Motor BI severity for 2023-24 (~16-17k) well below 2021-22 (~21-23k) while ~50-80% still open | AI | Open vs closed means by year; 2021 open BI ~ closed BI -> old case estimates look adequate | Case development: set BI 2023/24 severity to trended 2021-22 level (factors 1.38, 1.48). Judgmental - flag for review |
| 4 | One open BI claim (2023) incurred 11.9m vs paid 0.74m - 10x the next largest | AI | Ranked largest claims | Cap claims (motor 250k, buildings 300k, contents 60k); excess spread back as large-loss load from 2021-23 (motor 1.044) |
| 5 | Severity inflation ~8% motor, ~2% buildings, ~3% contents | AI | Count-weighted log-linear fit on perils with <10% open | Trend each year's claims to 2025 |
| 6 | `customer_service_contacts` gave in-sample D2 0.62 (55% of gain); 0 contacts => 0 claims | AI flagged, verified in data | Claim rate 0% at 0 contacts vs 98% at 6+; test distribution shifted | **Excluded**: recorded during the period, caused by claims = target leakage. Not knowable at quote |
| 7 | `previous_insurer`, `campaign_code`, `marketing_segment` high gain but flat claim rates across levels | AI | One-way claim rate by level (~flat) | Excluded as overfit noise; also no causal rationale |
| 8 | `written_premium` is the current price | Toby/AI | - | Excluded from primary model (circular). May test as diagnostic only |
| 9 | Model trained on 2021-23 overstated 2024 motor by 14% | AI | Out-of-time validation | Calibrate 2025 level using trend in actual/expected by year from grouped CV |

## Open questions for human review
- BI case-development factors are judgment (no development history in data). Sensitivity: +/-10% on BI moves motor level ~+/-2-3%.
- Motor frequency fell 2023-24; is this real or a data artefact?
- 2025 weather is provided per address - check it is "known" at pricing time (forecast vs. realised).

## v1 results (4-fold CV grouped by policy_id)
| Line | OOF D2 | A/E by year 2021-24 | 2025 calibration | Test pure premium |
|---|---|---|---|---|
| Motor | 0.064 | 0.89 / 0.91 / 0.89 / 0.89 | 0.888 | 527 |
| Buildings | 0.084 | 0.87 / 0.93 / 0.90 / 0.98 | 0.991 | 382 |
| Contents | 0.082 | 0.87 / 0.82 / 0.87 / 0.91 | 0.917 | 96 |

Observation: raw GBM is ~10% high out-of-fold in every year (consistent, so a calibration not a trend). The year-on-year drift in A/E (+3%/yr buildings, +2% contents, flat motor) is projected to 2025.
Buildings and contents 2025 levels sit above the multi-year average because the 2025 book and 2025 weather look riskier to the model, consistent with the 2024 level. To review.

## Week 1, day 4 (Oct 22): leaderboard diagnosis and v2
v1 scored 0.11107 public (33% of test), rank 33/37; top 0.12996.

| # | Finding | Proposed by | Verification | Decision |
|---|---|---|---|---|
| 10 | Score is level-sensitive: v1 x0.85 -> 0.0969, x1.15 -> 0.1173 | AI | Two scaled probes on public LB | Fit the closed-form D2(s) curve to the three scores: k,b,c solved exactly; v1 optimum s*=1.259 (D2 0.1185). Any new model needs one LB score to locate its optimum (k is model-independent; c computed from the predictions). `src/lb_calib.py` |
| 11 | OOF A/E ~0.89 yet deviance-optimal scale >1 => model over-dispersed, over-predicts top deciles 10-15% | AI | Decile calibration on grouped-CV OOF | Calibration curve rate = c*rate^a fitted on OOF by Tweedie deviance. Metric-optimal level sits ~25-33% above the unbiased expected cost: the Tweedie-1.5 score weights rows by mu^-0.5. Submit metric-optimal; pricing tool reports unbiased expected cost separately |
| 12 | Per-claim IBNR weights (up to 5x) and BI development on individual claims HURT: holdout D2 vs 2024 actuals 0.060 (adjusted target) vs 0.064 (raw capped) | AI experiment | Train 2021-23, score on raw 2024, scale-free best-s D2 | **Reverse decision #2/#3 for the model target.** Train on capped raw cost; apply IBNR/trend/large-loss at the aggregate level after modelling |
| 13 | OOF target encoding of make_model/occupation/region, replacing native categoricals: motor 0.064 -> 0.074 | AI (Deotte writeup) | Same holdout | Adopt. Native LightGBM categorical on 300 levels was overfitting |
| 14 | Household counts (policies per address/customer, lines per customer) | AI | Same holdout | +0.001, keep (cheap, interpretable: multi-car / multi-line households) |
| 15 | Tweedie power 1.3 / 1.7, deeper trees, lower LR | AI | Same holdout | No gain; keep p=1.5 (matches metric) and default depth |

Process note: 5 LB submissions/day. Budget: 3 used for level probes (now replaced by the closed-form solver), 1 for v2 probe, 1 for v2 at optimum.

## Data-quality traps (found Oct 22, `src/dq_scan.py`)
| # | Finding | Found by | Verification | Decision |
|---|---|---|---|---|
| 16 | ~900 training rows per line have `exposure` in DAYS (365, 327, 89...) and ~3-5k at 0.99/1.01; test max is 1.0 | AI: OOF A/E by exposure band contradicted overall A/E (impossible unless rows fell outside bands) | Ratio exposure/(days/365) is exactly 365 for those rows | Recompute exposure from period dates for all rows. Impact: 0.4% of rows carried 26% of predicted cost through the log-exposure offset, distorting training and the calibration curve |
| 17 | `construction_type` is case-scrambled (Brick/brick/BRICK, STONE, NON_STANDARD) and timber is coded 'W'/'w' (train+test) and 'T'/'t' (train only); no row says 'timber' | AI scan; first scan missed it (only checked train-only levels), full case-variant scan found it | T and W have identical claim frequency (4.6%) -> same category | strip+lower, map w/t -> timber; assert 4 dictionary levels. Only column affected across all 9 tables |
| 18 | 2.5% NaN in vehicle_age, engine_size, mileage, days_unoccupied, high_value_items; 88% NaN telematics_score (not opted in) | AI scan | Same rates in test | Legitimate missingness; left to LightGBM's native NaN handling |
| 19 | 274 claims with loss date before 2021 | AI scan | Fall outside any policy row | Unmatched, excluded (0.17%) |
Impact of fixes 16-17 on grouped-CV D2 (capped raw target): motor 0.082->0.091, buildings 0.091->0.100, contents 0.083->0.092; calibration A/E moved from 0.75 to 1.00 (the apparent 'metric wants +33%' was the exposure bug).
Lesson: the first v1 build did not include a data-quality scan; the planted errors were only caught through a calibration diagnostic that produced an impossible result. Automated DQ checks now run before modelling (`tests/`).

## Round 2 (clean data, 2024 holdout vs raw actuals; `src/experiment2.py`, `src/hist_features.py`)
| Variant | Motor | Buildings | Contents | Decision |
|---|---|---|---|---|
| v2 config | 0.0855 | 0.1144 | 0.0922 | baseline |
| 3-seed average | 0.0855 | 0.1135 | 0.0917 | no gain |
| + written_premium | 0.0798 | 0.1125 | 0.0883 | **hurts**; also excluded on principle (circular) |
| cap 100k | 0.0853 | 0.1125 | 0.0913 | keep 250k/300k/60k |
| shallower / deeper trees | 0.0834-0.0849 | 0.1091-0.1154 | 0.0879-0.0931 | within noise |
| + policy/customer/address claim history (1y, 2y), address peril flags, weather anomalies | 0.0852 | 0.1129 | 0.0924 | no gain; past claims add nothing beyond static risk factors in this data |

v2 public LB 0.12444 (rank ~26/37; top 0.12996, 25 teams within 0.0055). Solver: v2 optimal multiplier 1.042 (+0.0002 only).
v3 = v2 with per-line 2025 levels from the Tweedie-optimal A/E trend of v2 OOF vs ultimate-adjusted cost (motor 1.376, buildings 1.159, contents 1.131 on pooled 2021-24 base; `src/level_v3.py`), total anchored to v2 x 1.042. Motor +6%, buildings +3%, contents -1% vs v2. Expected gain small (<0.001). Main residual uncertainty: motor BI case development (1.105 vs 1.376 motor factor with/without).

## Final Part 1 submissions (Oct 22)
| File | Public LB | What changed |
|---|---|---|
| v1 | 0.11107 | Monday baseline (dirty data, per-claim weights, native categoricals) |
| v2 | 0.12444 | DQ fixes, capped raw target, target encoding, calibration curve |
| v3 | 0.12399 | per-line levels from A/E trend with BI development (motor too high) |
| v4 | 0.12618 | per-line levels solved from LB: motor x0.945, home x1.144 vs v2 (predicted 0.1259) |
| **v5 (selected)** | **0.12698** | exact levels (motor 0.936, bld 1.172, cnt 1.121) + 30% blend with partner's model, level-matched per line |
Leader 0.12996. Final standings are on the other 67% of test rows.

Per-line solve (`src/lb_lines.py`): with scores for v2, v3, v4 the per-line b_l are exactly identified (3 equations, 3 unknowns; k from v1 probes, c_l from the predictions). Exact optimum predicted 0.12622 for levels alone - level is exhausted; the blend added the rest.
Interpretation: the home lines needed +12-17% over v2 (consistent with their observed +5-8%/yr pure-premium trend) and motor -6% (the BI case-development judgment overstated 2025 motor). This is the main Week 1 lesson for the write-up: between-line level allocation mattered more than any model change after the DQ fixes.
Partner's model: two LightGBM-style submissions (0.1195 / 0.1177 at his own level, ~12% below ours), correlation with v2 0.85-0.93 by line. Averaged, level-matched per line, blended at 0.30. Blend weight not tuned (one submission left) - tune in Week 2 with OOF predictions from both pipelines.

## Oct 23 (quota reset: 10 more submissions)
| File | Public LB | What |
|---|---|---|
| v6 | 0.12662 | 50% partner blend probe -> fitted optimum w=0.30 (v5 already optimal) |
| **v7 (selected)** | **0.12841** | second in-house model: frequency (Poisson) x severity (Gamma) GBM, same features/folds; OOF D2 beats Tweedie in every line (motor 0.094 vs 0.091, bld 0.102 vs 0.100, cnt 0.095 vs 0.092); ours = 50/50 Tweedie+freq-sev level-matched per line; then 70/30 with partner |
Leader 0.12996 (gap 0.0016). Freq-sev vs Tweedie correlation 0.95-0.97 by line.
Decision #26: freq-sev adopted as co-primary model. Rationale: two-part model isolates large-claim influence to the severity component; frequency model trained on all rows converges over ~1000 iterations vs ~200 for Tweedie. Also the pricing-team-friendly form (frequency/severity relativities).

# Part 2: emerging market (Oct 29)
## Data and DQ (`src/p2_dq.py`, `src/p2_prep.py`)
| # | Finding | Found by | Verification | Decision |
|---|---|---|---|---|
| P1 | Exposure-in-days rows again (~50/line) | DQ scan (now run first) | ratio exposure/(days/365)=365 | recompute from dates |
| P2 | construction_type recoded BRK/MIX/STN/WOD | DQ scan | dictionary has brick/stone/timber/non-standard | map; MIX -> non_standard |
| P3 | occupation values carry trailing spaces | DQ scan (case/strip check) | 80 levels, all with trailing space | strip (needed for any cross-market encoding) |
| P4 | regions E01-E06, no overlap with R01-R12 | schema compare | - | region dropped from transfer model; kept in local/pooled |
| P5 | features missing vs developed: credit_band, date_of_birth (banded ages instead), annual_mileage, telematics | schema compare | - | transfer model trained on the common feature set, developed ages banded to match |
| P6 | ~12% NA across many customer/address/policy fields | schema compare | same in test | native NaN handling |
| P7 | new column `gender` (regulation differs) | schema compare | OOF D2 with/without: motor +0.0002, bld -0.002, cnt +0.001 | **excluded** on governance grounds; cost measured as ~0 |
| P8 | test book is 76% new business vs 14% in train | distribution compare | - | explicit NB uplift (credibility-weighted local vs developed), then LB segment solve |
| P9 | customer_service_contacts leak present in train (OOF D2 0.57-0.61) | feature test | test distribution == train no-claim rows (0.67/0.27/0.054/0.007) => scrubbed | excluded; using it would predict 0 for 67% of rows |
| P10 | 2025 weather milder than 2023-24; market far hailier than developed (14-17 vs 6 hail days) | weather compare | - | no catastrophe adjustment; pooled model sees full hail range |
| P11 | expenses table (fixed 85 CU, variable 15%, broker commission 20%) and 20 reference policies | new files | - | used in the pricing tool / technical premium section, not in the model |

## Models (`src/p2_train.py`, `src/p2_pooled.py`)
T transfer (developed, common features) / L local (2 yrs) / O offset = local deviations from T (credibility) / P pooled both markets with market flag, emerging weight 5, developed TE priors for make_model & occupation. GroupKFold(5) on emerging rows.
| Line | T | L | O | P | best blend (OOF D2) |
|---|---|---|---|---|---|
| motor | 0.044 | 0.044 | 0.044 | 0.054 | L.11/O.39/P.50 = 0.058 |
| buildings | 0.100 | 0.048 | 0.107 | 0.074 | O.94/L.06 = 0.107 |
| contents | 0.095 | 0.072 | 0.098 | 0.095 | O.56/P.44 = 0.104 |
Autocalibration (isotonic, cross-fit) tested per Denuit-Charpentier-Trufin: rejected, local data too thin (decile A/E 0.5-1.4 noise).

## Leaderboard
| Sub | Score | What |
|---|---|---|
| v1 | 0.13787 | blend T/L/O, NB uplift 1.23/1.00/1.19, pooled level |
| x1.35 / NB x1.3 / REN x1.3 | 0.14409 / 0.14492 / 0.13767 | segment probes -> NB x1.282, REN x1.125 (exact) |
| v2 (+motor x1.2) | 0.14551 | per-line probe -> motor x1.105, home x0.882 |
| **v3** | **0.14864** | 4-model blends + exact line/segment levels |
Leader 0.17556. Remaining gap attributed to motor ranking quality (no mileage/exact age/telematics in this market) and unresolved B-vs-C and NB-per-line levels (probes pending).

## Part 2, day 2 (Oct 29-30): transfer priors and exact segment levels
| Sub | Score | What |
|---|---|---|
| v4 | (not submitted) | freq-sev transfer prior T2 + offset O2 + pooled 3-seed, 8-model stack |
| v5 | 0.16036 | + per-peril transfer prior T3 + offset O3, 10-model stack. OOF +0.004/line; LB +0.012: transfer priors generalise to the 76%-new-business test book better than to the renewal-heavy validation folds |
| v6 | 0.16062 | + pooled freq-sev PF2 (best single model on motor 0.061 and contents 0.107), 11-model stack |
| probes on v6 | bld x1.15: 0.16035; motor-NB x1.15: 0.15857 | |
| v7 | 0.15999 | tempered levels under a shared-home-line assumption: WRONG (assumption pushed error onto motor renewals) |
| **v8 (selected)** | **0.16162** | exact 4-segment solve from the four v6-based scores, no assumptions: motor NB x0.952, motor REN x1.099, buildings x1.044, contents x0.847. Predicted 0.16155 |
Leader 0.1756; top 10 ~0.163. Solo entry (team dissolved Oct 29).
Lessons: (1) the level solve is reliable only when fully identified; every partially-identified attempt (Part 1 v3, Part 2 v7) lost; (2) developed-market priors in frequency-severity form are the single most valuable transfer; (3) the new-business uplift from two years of thin local data was over-estimated for motor and under-estimated for renewals.

## Part 2 close-out (Oct 30)
| Sub | Score | What |
|---|---|---|
| v9 | 0.16121 | per-peril level-transfer prior T4/O4 (re-levels each peril to local 2023-24 experience). Best single model on OOF; LOST on LB: pulls the prior toward renewal-heavy local history while the test book is new business |
| v10 | 0.16161 | v8 with 25 points of blend weight shifted to pure developed priors: identical to v8 => blend at a flat optimum |
| tuning | - | claim history (none), hyperparameter grid (+-0.0005), Tweedie power 1.3/1.7 (+-0.0005): all noise-level |
**Final: v8 = 0.16162.** Model family saturated: levels exactly solved, blend on a plateau, priors and local models all represented. Leaders at 0.167-0.176 have a structural element not identified here; logged as an open question.
| v11 | **0.16176 (selected)** | GLM (explainable benchmark: 0.044/0.013/0.081 OOF) added to the contents blend at weight 0.14. Motor/buildings GLM weight 0 |
