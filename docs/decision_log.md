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
