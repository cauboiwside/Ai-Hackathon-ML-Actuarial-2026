"""Assemble a submission from components, all level-matched per line to the solved line totals.
Components per line: tw (v2 Tweedie, calibrated), fs (freq-sev, calibrated), pp (per-peril, if available), his (partner)."""
import sys, json, pandas as pd, numpy as np
W='/home/claude/iaa-pricing/work/'
S={'M':0.936,'B':1.172,'C':1.121}
def assemble(weights, out, w_his=0.1):
    v2=pd.read_csv(W+'submission_v2.csv',dtype={'row_id':str}).set_index('row_id').prediction
    line=v2.index.str[0]; base=v2*line.map(S).values
    fs=pd.read_parquet(W+'fs_cal_matched.parquet').set_index('row_id').prediction.reindex(v2.index)
    his=pd.read_parquet(W+'his_matched.parquet').set_index('row_id').prediction.reindex(v2.index)
    pp=pd.Series(np.nan,index=v2.index)
    for ln,L in [('motor','M'),('buildings','B'),('contents','C')]:
        try: p=pd.read_csv(W+f'pred_fsp_{ln}.csv',dtype={'row_id':str}).set_index('row_id').prediction; pp[p.index]=p.values
        except FileNotFoundError: pass
    ours=base.copy()
    for L in 'MBC':
        m=line==L; w=weights[L]; tot=base[m].sum()
        comp=w['tw']*base[m]+w['fs']*fs[m]
        if w.get('pp',0)>0:
            q=pp[m]; assert q.notna().all(), f'per-peril missing for {L}'; comp=comp+w['pp']*q*tot/q.sum()
        ours[m]=comp*tot/comp.sum()
    final=(1-w_his)*ours+w_his*his
    assert (final>=0).all() and np.isfinite(final).all() and len(final)==626639
    final.rename('prediction').reset_index().to_csv(out,index=False); print('wrote',out,weights)
    return final
if __name__=='__main__':
    assemble(json.loads(sys.argv[1]),sys.argv[2])
