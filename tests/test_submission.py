"""Automated guardrails on the submission file. Run: python -m pytest tests/"""
import pandas as pd, numpy as np, json, pathlib
W=pathlib.Path('/home/claude/iaa-pricing/work'); P=pathlib.Path('/home/claude/data_raw/public')
SUB=sorted(W.glob('submission_v*.csv'))[-1]
s=pd.read_csv(SUB,dtype={'row_id':str}); ss=pd.read_parquet(P/'sample_submission.parquet')
def test_columns(): assert list(s.columns)==['row_id','prediction']
def test_all_rows_once(): assert s.row_id.is_unique and set(s.row_id)==set(ss.row_id)
def test_non_negative_finite(): assert np.isfinite(s.prediction).all() and (s.prediction>=0).all()
def test_level_reasonable():
    r=json.load(open(W/'cv_results.json'))
    for ln,v in r.items():  # test pure premium within 25% of the 2025-level training pure premium
        assert 0.75 < v['test_pp']/v['train_pp'] < 1.25, ln
def test_no_extreme_rows():  # no row above 100x its line mean
    s['line']=s.row_id.str[0]
    m=s.groupby('line').prediction.transform('mean'); assert (s.prediction < 100*m).all()
