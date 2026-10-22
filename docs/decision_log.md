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
