"""Per-line 2025 level: Tweedie-optimal A/E of v2 OOF vs ultimate-adjusted target by year, trended to 2025.
Global level anchored to the LB-optimal v2 total (s*=1.042); per-line factors redistribute between lines."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from common import *
cl=pd.read_parquet(W+'claims_level.parquet'); cv=json.load(open(W+'cv_results_v2.json')); LB_S=1.042
full=pd.read_parquet(W+'pred_v2_full.parquet'); full['line']=full.row_id.str[0].map({'M':'motor','B':'buildings','C':'contents'})
te=pd.concat([pd.read_parquet(W+f'mf_test_{ln}.parquet',columns=['row_id','exposure']) for ln in cv]); full=full.merge(te,on='row_id')
fac={}
for ln in cv:
    o=pd.read_parquet(W+f'oof_v2_{ln}.parquet')
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(cl,on='claim_id')
    for nm,col in [('ibnr',j.cost_c*j.w),('ult',j.cost_c*j.w*j.dev)]: o['y_'+nm]=o.row_id.map(col.groupby(j.row_id).sum()).fillna(0)
    # metric-optimal per-year scale of calibrated OOF (apply v2 calibration curve first)
    a,c=cv[ln]['calib_a'],cv[ln]['calib_c']; o['cal']=o.exposure*c*(o.oof/o.exposure)**a
    rows=[]
    for yr,g in o.groupby('year'):
        s={nm:minimize_scalar(lambda s: tweedie_dev(g['y_'+nm].values,s*g.cal.values).sum(),bounds=(0.3,3),method='bounded').x for nm in ['ibnr','ult']}
        rows.append((yr,s['ibnr'],s['ult']))
    t=pd.DataFrame(rows,columns=['year','s_ibnr','s_ult']).set_index('year'); print(ln); print(t.round(3).T)
    b1,b0=np.polyfit(t.index,np.log(t.s_ult),1); s25=np.exp(b0+b1*2025)
    b1i,b0i=np.polyfit(t.index,np.log(t.s_ibnr),1); s25i=np.exp(b0i+b1i*2025)
    print(f'  A/E trend (ult) {np.exp(b1)-1:+.1%}/yr -> 2025 factor {s25:.3f}   [ibnr-only: {s25i:.3f}]   v2 used {cv[ln]["level"]:.3f}')
    fac[ln]=dict(s2025_ult=float(s25),s2025_ibnr=float(s25i),ll=lp if (lp:=json.load(open(W+'level_params.json'))['ll'][ln]) else 1)
# build v3: metric-optimal v2 prediction / v2 level * new per-line level (ult) * large-loss load, then rescale total to v2*LB_S
full['base']=full.prediction/full.line.map({k:v['level'] for k,v in cv.items()})
full['v3']=full.base*full.line.map({k:v['s2025_ult']*v['ll'] for k,v in fac.items()})
tot=full.prediction.sum()*LB_S; full['v3']*=tot/full.v3.sum()
print('\nper-line pp (per exposure-year): v2 vs v3'); print(full.groupby('line').apply(lambda g: pd.Series({'v2':g.prediction.sum()/g.exposure.sum(),'v3':g.v3.sum()/g.exposure.sum()}),include_groups=False).round(1))
full[['row_id']].assign(prediction=full.v3).to_csv(W+'submission_v3.csv',index=False); json.dump(fac,open(W+'level_v3.json','w'),indent=1)
