# IAA AI Assisted Hackathon 2026

**Team: Evan Tran & Toby Nguyen**

**Members: Evan Tran and Toby Nguyen**

Insurance claim-cost predictions for motor, home buildings and home contents, adapted from a developed to an emerging market. AI assisted exploration, implementation, diagnostics and documentation. The existing Git history and decision log preserve the development process.

## Final submission artifacts

| Part | Canonical file | Rows | Team confirmed public score |
| --- | --- | ---: | ---: |
| Week 1 | [part1_prediction_file.csv](final_submissions/part1_prediction_file.csv) | 626,639 | 0.12944 |
| Week 2 | [part2_prediction_file.csv](final_submissions/part2_prediction_file.csv) | 209,063 | 0.16515 |

The selected sources are `submission_v14.csv` and `p2_submission_v17_FINAL.csv`. Both canonical copies match these sources byte for byte. Each was checked against its official sample and all three test-policy tables: exact `row_id,prediction` columns, every required ID once in sample order, and positive finite predictions. See the [audit](docs/prediction_audit.md), [validation report](docs/submission_validation.json) and [hash manifest](final_submissions/manifest.json).

The earlier Part 1 repository copy differed at every prediction row despite identical line totals. The selected v14 is now canonical; the earlier version remains in Git history. No source prediction values were changed. File integrity does not establish predictive accuracy. Scores above were confirmed by the team; no private score or final rank is claimed.

## Read and record

- [Revised four-page write-up](docs/IAA_Hackathon_Writeup_Evan_Tran_and_Toby_Nguyen.docx) and [Markdown version](docs/writeup.md)
- [Demo recording script](docs/video_script.md), designed for less than five minutes
- [Submission checklist](docs/SUBMISSION_CHECKLIST.md)
- [Reproduction instructions and outstanding gaps](docs/REPRODUCING.md)
- [Historical decision log](docs/decision_log.md) and [Part 2 experiment notes](docs/part2_process.md)
- [Review corrections](docs/REVIEW_NOTES.md)

## Validate the files

CSV checks need only Python's standard library:

```sh
python3 scripts/validate_submissions.py --report /tmp/iaa-validation.json
python3 -m unittest discover -s tests -v
```

For exact ID checks against the official parquet samples, install `requirements.txt` in an isolated environment and run:

```sh
python3 scripts/validate_submissions.py \
  --part1-sample /path/to/public/sample_submission.parquet \
  --part2-sample /path/to/emerging/sample_submission.parquet \
  --report /tmp/iaa-validation.json
```

The review ran 9 tests successfully; one generated-frame test class was skipped because model intermediates are absent. Tests do not retrain models or independently verify Kaggle results.

## Models and evaluation

Part 1 combines Tweedie LightGBM, frequency-severity and per-peril models. Part 2 compares transfer, local, offset and pooled approaches on harmonised features. The final solution uses public-leaderboard feedback, including rating-segment calibration. Public results are adaptive tuning results, not independent validation. Historical target encoding and blend selection also limit interpretation of cross-validation scores; see the write-up.

This is an experiment archive with verifiable final artifacts. A complete end-to-end recipe for retraining and assembling exact v14/v17 predictions is still incomplete. Missing intermediates, final adjustment recipes and legacy paths are detailed in `docs/REPRODUCING.md`. Competition datasets are excluded; obtain them through the organiser's authorised distribution.
