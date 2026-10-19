# IAA AI-Assisted Hackathon 2026 - pricing model

Pipeline (run in order):
1. `src/prep.py` - attach claims to policy rows (policy_id + loss_date in period)
2. `src/level.py` - peril IBNR, large-loss cap & load, case development, severity trend -> claims at 2025 ultimate level
3. `src/features.py` - join customers, addresses, weather (by year), prior-year claim history
4. `src/train.py` - LightGBM Tweedie (p=1.5, log-exposure offset), 4-fold CV grouped by policy, A/E-by-year calibration, test predictions
5. `pytest tests/` - submission guardrails

Decisions and their verification: `docs/decision_log.md`.
