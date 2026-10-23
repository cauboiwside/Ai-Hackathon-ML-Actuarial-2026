"""Blend at exact per-line levels: base = v2 * S_l; partner = avg(his two files) level-matched per line; w can be per line."""
import sys, json, pandas as pd, numpy as np
W='/home/claude/iaa-pricing/work/'; S={'M':0.936,'B':1.172,'C':1.121}
def build(w, out, extra=None, wx=0.0):
    """w: dict per line or float. extra: optional third prediction series (level-matched), weight wx."""
    v2=pd.read_csv(W+'submission_v2.csv',dtype={'row_id':str}).set_index('row_id').prediction
    line=v2.index.str[0]; base=v2*line.map(S).values
    his=sum(pd.read_csv(f'/home/claude/partner/{f}.csv',dtype={'row_id':str}).set_index('row_id').prediction.reindex(v2.index) for f in ['risk_v1','risk_v2'])/2
    wl=w if isinstance(w,dict) else {l:w for l in 'MBC'}
    out_s=base.copy()
    for l in 'MBC':
        m=line==l; h=his[m]*base[m].sum()/his[m].sum()
        b=(1-wl[l]-wx)*base[m]+wl[l]*h
        if extra is not None: e=extra.reindex(base.index)[m]; b=b+wx*e*base[m].sum()/e.sum()
        out_s[m]=b
    assert (out_s>=0).all() and np.isfinite(out_s).all() and len(out_s)==626639
    out_s.rename('prediction').reset_index().to_csv(out,index=False); print('wrote',out,'w=',wl,'wx=',wx)
if __name__=='__main__':
    w=json.loads(sys.argv[1]); build(w if isinstance(w,dict) else float(w), sys.argv[2])
