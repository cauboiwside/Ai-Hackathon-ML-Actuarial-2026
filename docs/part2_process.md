# Part 2 process: 0.138 -> 0.162 (for anyone picking this up)

**Setup.** Emerging market, 2023-24 history (~5% of Part 1 volume); 2025 test is 2x the history and 76% new business from a broker. Missing vs developed: credit band, exact ages (banded), mileage, telematics; new region codes; new `gender` column; 12% NA.

**Step 0, DQ scan first** (`src/p2_dq.py`, fixes in `src/p2_prep.py`): exposure in days (~50 rows/line), construction_type BRK/MIX/STN/WOD, trailing spaces on occupation. `customer_service_contacts` leak verified scrubbed in test (distribution == training no-claim rows): excluded. Gender measured (+-0.001 D2): excluded on regulatory grounds.

**Step 1, four ways to use the developed market** (`src/p2_train.py`, `src/p2_pooled.py`), validated by GroupKFold(5) on emerging rows:
T transfer (developed model, common features, re-levelled) / L local only / O offset = local model on deviations from T (credibility) / P pooled both markets with market flag, emerging weight 2.
Home lines: prior worth 2x local data (0.100 vs 0.048). Motor weak everywhere (~0.044).
v1 = 0.1379: T/L/O blend + new-business uplift from local evidence.

**Step 2, level solve** (`src/p2_lb_solve.py`): 1-D2 = k + sum_seg(b s^-0.5 + c s^0.5); k truth-only, c from predictions, b from scores. Three probes -> NB x1.282, REN x1.125 (v2 = 0.1455, with motor x1.2 probe -> per-line). v3 = 0.1486: 4-model blend + exact levels.

**Step 3, frequency x severity priors** (`src/p2_improve.py`, `src/p2_peril.py`, `src/p2_pooled_fs.py`): T2 freq-sev transfer, T3 per-peril transfer, O2/O3 offsets (5 seeds), PF2 pooled freq-sev. Held-out +0.004/line; LB v5 = 0.1604 (+0.012: the new-business test book is described better by developed relativities than by the renewal-heavy local history). v6 = 0.1606 with PF2.

**Step 4, exact segment levels on v6.** Probes bld x1.15 (0.16035) and motor-NB x1.15 (0.15857). v7 (0.15999) used one closing assumption (home lines share a level) and lost; with v7 as a 4th equation the 4 segments are exactly identified: motor NB x0.952, motor REN x1.099, buildings x1.044, contents x0.847. v8 = 0.1616 (predicted 0.16155).

**Lessons.** DQ scan first. Freq-sev priors from the rich market are the best transfer. The level solve is exact only when fully identified. The pricing tool's finding (technical premium 2-6x the broker's written premium; `src/pricing_tool.py`) matters more commercially than any 0.001 of D2.

**Repo map.** `src/p2_*.py` in the order above; `work_p2/oof_<line>.parquet` and `pred_<line>.parquet` hold every model's OOF/test predictions; `work_p2/stack_p2_v*.json` the blend weights; `src/p2_build_v3.py` assembles a submission from weights + segment multipliers; `docs/decision_log.md` every decision with its verification.
