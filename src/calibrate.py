"""Fit per-line calibration curve rate_cal = c * rate^a on OOF (Tweedie deviance), apply to test preds."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import *
cv=json.load(open(W+'cv_results.json'))
sub=pd.read_csv(W+'submission_v1.csv',dtype={'row_id':str})
te=pd.concat([pd.read_parquet(W+f'mf_test_{ln}.parquet',columns=['row_id','exposure']).assign(line=ln) for ln in cv])
sub=sub.merge(te,on='row_id')
out=[]; params={}
for ln in cv:
    o=pd.read_parquet(W+f'oof_{ln}.parquet'); lr=np.log(o.oof/o.exposure); e=o.exposure.values; y=o.y.values
    f=lambda p: tweedie_dev(y,e*np.exp(p[0]+p[1]*lr)).sum()
    r=minimize(f,[0,1],method='Nelder-Mead'); c,a=r.x
    mu2=e*np.exp(c+a*lr)
    print(f'{ln}: a={a:.3f} c={np.exp(c):.3f}  OOF D2 {d2(y,o.oof.values):.4f} -> {d2(y,mu2):.4f}  A/E {y.sum()/mu2.sum():.3f}')
    s=sub[sub.line==ln]; raw_rate=s.prediction/(cv[ln]['cal_2025']*cv[ln]['ll'])/s.exposure  # undo v1 level factors
    # keep the year-drift part of the 2025 calibration relative to pooled OOF level, and the large-loss load
    drift=cv[ln]['cal_2025']/np.exp(np.polyfit(list(map(int,cv[ln]['ae'].keys())),np.log(list(cv[ln]['ae'].values())),0)[0])
    new=s.exposure*np.exp(c+a*np.log(raw_rate))*cv[ln]['ll']*drift
    print(f'   drift={drift:.3f} test pp v1={s.prediction.sum()/s.exposure.sum():.1f} -> v1b={new.sum()/s.exposure.sum():.1f}')
    out.append(pd.DataFrame({'row_id':s.row_id,'prediction':new.values})); params[ln]=dict(a=float(a),c=float(np.exp(c)),drift=float(drift))
pd.concat(out).to_csv(W+'submission_v1b.csv',index=False); json.dump(params,open(W+'calib_v1b.json','w'),indent=1)
