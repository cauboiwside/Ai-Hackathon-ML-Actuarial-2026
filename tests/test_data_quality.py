"""Pre-modelling data-quality guardrails on the model frames."""
import pandas as pd, pathlib
W=pathlib.Path('/home/claude/iaa-pricing/work')
def test_exposure_in_years():
    for ln in ['motor','buildings','contents']:
        for s in ['train','test']:
            e=pd.read_parquet(W/f'mf_{s}_{ln}.parquet',columns=['exposure']).exposure
            assert e.between(0.002,1.005).all(), (ln,s,e.max())
def test_categorical_levels_match():
    for ln in ['motor','buildings','contents']:
        tr=pd.read_parquet(W/f'mf_train_{ln}.parquet'); te=pd.read_parquet(W/f'mf_test_{ln}.parquet')
        for c in tr.columns:
            if str(tr[c].dtype) in ('object','str') and c in te and not c.endswith('_id'):
                assert set(te[c].dropna())<=set(tr[c].dropna()), (ln,c)
