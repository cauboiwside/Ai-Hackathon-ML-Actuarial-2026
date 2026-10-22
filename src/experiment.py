"""Experiment loop: train 2021-23, evaluate on 2024 RAW observed cost (scale-free best-s D2) + at s=1."""
import sys, json, time; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
P='/home/claude/data_raw/public/'
LN=sys.argv[1]
BASE=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,
         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,
         cat_l2=10,verbose=-1,num_threads=2,seed=1)
d=pd.read_parquet(W+f'mf_train_{LN}.parquet')
raw=pd.read_parquet(W+f'train_{LN}.parquet',columns=['row_id','cost'])
d=d.merge(raw,on='row_id')
cl=pd.read_parquet(W+'claims_level.parquet'); cr=pd.read_parquet(W+f'claimrows_{LN}.parquet',columns=['claim_id','row_id'])
j=cr.merge(cl,on='claim_id')
# alt targets
capped=j.groupby('row_id').cost_c.sum(); d['t_cap']=d.row_id.map(capped).fillna(0)
trended=(j.cost_c*j.trend).groupby(j.row_id).sum(); d['t_captrend']=d.row_id.map(trended).fillna(0)
d['t_adj']=d.target
# household features
pm=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['row_id','policy_id','customer_id','address_id']).assign(l=f[-1]) for f in ['policies_motor','policies_buildings','policies_contents']])
d['n_pol_addr']=d.address_id.map(pm.groupby('address_id').policy_id.nunique())
d['n_cust_addr']=d.address_id.map(pm.groupby('address_id').customer_id.nunique())
d['n_pol_cust']=d.customer_id.map(pm.groupby('customer_id').policy_id.nunique())
d['n_lines_cust']=d.customer_id.map(pm.groupby('customer_id').l.nunique())
tr=d.year<=2023; va=d.year==2024
def te_encode(df,cols,target,trmask,k=5,prior_w=200):
    """OOF target encoding (rate per exposure) on training rows; full-train encoding for others."""
    out=pd.DataFrame(index=df.index)
    g=df[trmask]; mean=g[target].sum()/g.exposure.sum()
    for c in cols:
        enc=np.full(len(df),np.nan)
        for a,b in GroupKFold(k).split(g,groups=g.policy_id):
            s=g.iloc[a].groupby(c,observed=True).agg(y=(target,'sum'),e=('exposure','sum'))
            r=(s.y+prior_w*mean)/(s.e+prior_w)
            enc[np.where(trmask)[0][b]]=g.iloc[b][c].map(r).fillna(mean).values
        s=g.groupby(c,observed=True).agg(y=(target,'sum'),e=('exposure','sum')); r=(s.y+prior_w*mean)/(s.e+prior_w)
        enc[~trmask.values]=df[~trmask][c].map(r).fillna(mean).values
        out['te_'+c]=np.log(enc/mean)
    return out
def run(name,target,extra=None,params=None,drop=None,te=None):
    t0=time.time(); X,cats=prep(d.drop(columns=['cost','t_cap','t_captrend','t_adj','n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']))
    if extra is not None: X=pd.concat([X,d[extra]],axis=1)
    if te: X=pd.concat([X,te_encode(d,te,target,tr)],axis=1)
    if drop: X=X.drop(columns=drop); cats=[c for c in cats if c not in drop]
    prm={**BASE,**(params or {})}
    y=d[target].values; off=np.log(d.exposure.values)+np.log(y[tr].sum()/d.exposure[tr].sum())
    ds=lgb.Dataset(X[tr],y[tr],init_score=off[tr],categorical_feature=cats)
    dv=lgb.Dataset(X[va],y[va],init_score=off[va],categorical_feature=cats,reference=ds)
    m=lgb.train(prm,ds,4000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
    mu=np.exp(m.predict(X[va],raw_score=True)+off[va]); yr=d.cost[va].values
    best=max((d2(yr,mu*s),s) for s in np.arange(0.5,1.5,0.025))
    print(f'{name:38s} it={m.best_iteration:4d} D2raw(s=1)={d2(yr,mu):.4f} best={best[0]:.4f} @s={best[1]:.3f} pred_pp={mu.sum()/d.exposure[va].sum():.0f} {time.time()-t0:.0f}s',flush=True)
HH=['n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']
TE=['make_model','occupation','region'] if LN=='motor' else ['occupation','region']
import os
if os.environ.get('QUICK'):
    run('B capped raw target','t_cap'); run('F cap + TE, drop raw cats','t_cap',te=TE,drop=TE); run('J cap + HH + TE drop','t_cap',extra=HH,te=TE,drop=TE); run('K J + trend target','t_captrend',extra=HH,te=TE,drop=TE); sys.exit()
run('A baseline adjusted target','t_adj')
run('B capped raw target','t_cap')
run('C capped+trended target','t_captrend')
run('D cap + household feats','t_cap',extra=HH)
run('E cap + target enc','t_cap',te=TE)
run('F cap + TE, drop raw cats','t_cap',te=TE,drop=TE)
run('G cap + HH + TE + tuned','t_cap',extra=HH,te=TE,params=dict(learning_rate=0.02,num_leaves=63,min_data_in_leaf=300,lambda_l2=30))
run('H cap + HH + TE + p1.3','t_cap',extra=HH,te=TE,params=dict(tweedie_variance_power=1.3))
run('I cap + HH + TE + p1.7','t_cap',extra=HH,te=TE,params=dict(tweedie_variance_power=1.7))
