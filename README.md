# IAA AI-Assisted Hackathon 2026 — actuarial pricing across markets

Toby Nguyen. Part 1 (developed market) public LB 0.1294 (8th of 37, team entry); Part 2 (emerging market) public LB 0.1618 (solo).

**Read first:** `docs/writeup.md` (5 pages) · `docs/decision_log.md` (every decision, who proposed it, how it was verified) · `docs/part2_process.md`.

## Pipeline

Part 1 (`src/`, run in this order): `prep.py` (claims → policy rows) · `level.py` (IBNR, caps, trend) · `features.py` (joins + DQ fixes) · `train_v2.py` (Tweedie GBM, OOF target encoding) · `train_fs.py` (frequency × severity) · `train_fs_peril.py` (per-peril) · `build_final.py` (blend) · `lb_calib.py`, `lb_lines.py` (closed-form leaderboard level solve) · `dq_scan.py`, `experiment*.py`, `hist_features.py` (experiments, mostly negative, all logged).

Part 2 (`src/p2_*.py`): `p2_dq.py` · `p2_prep.py` (emerging frames + developed common-feature frames) · `p2_train.py` (transfer / local / offset) · `p2_pooled.py`, `p2_pooled_fs.py` (pooled two-market) · `p2_improve.py`, `p2_peril.py`, `p2_peril_level.py`, `p2_peril_cred.py` (frequency-severity and per-peril transfer priors, credibility offsets) · `p2_glm.py` (explainable benchmark) · `p2_lb_solve.py` (segment level solve) · `p2_build_v3.py` (assemble a submission from stack weights + segment multipliers) · `pricing_tool.py` (risk premium → technical premium; `docs/reference_policies_priced.csv`).

Tests: `pytest tests/` (submission format, exposure in years, categorical consistency, level sanity).

## Final files
`work/submission_part1_final_solo.csv` (Part 1), `work_p2/p2_submission_v13.csv` (Part 2). Data is not in the repo (Kaggle licence); place the Part 1 parquet files in `/home/claude/data_raw/public/` and the Part 2 `emerging/` folder in `/home/claude/data_p2/` to reproduce.
