"""Round 2 experiments on CLEAN data: train 2021-23, score vs raw 2024 (scale-free)."""
import sys, time; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
P='/home/claude/data_raw/public/'; LN=sys.argv[1]
BASE=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,
         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=1)
TE={'motor':['make_model','occupation','region'],'buildings':['occupation','region'],'contents':[]}[LN]
HH=['n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']; useHH=LN=='motor'
import os
d=pd.read_parquet(W+(f'mfh_train_{LN}.parquet' if os.environ.get('HIST') else f'mf_train_{LN}.parquet')); cl=pd.read_parquet(W+'claims_level.parquet')
j=pd.read_parquet(W+f'claimrows_{LN}.parquet',columns=['claim_id','row_id']).merge(cl,on='claim_id')
d['t_cap']=d.row_id.map(j.groupby('row_id').cost_c.sum()).fillna(0); d['raw']=d.row_id.map(j.groupby('row_id').cost.sum()).fillna(0)
d['t_cap100']=d.row_id.map(j.assign(c=j.cost.clip(upper=100_000)).groupby('row_id').c.sum()).fillna(0)
pm=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['row_id','policy_id','customer_id','address_id']).assign(l=f[-1]) for f in ['policies_motor','policies_buildings','policies_contents','test_motor','test_buildings','test_contents']])
d['n_pol_addr']=d.address_id.map(pm.groupby('address_id').policy_id.nunique()); d['n_cust_addr']=d.address_id.map(pm.groupby('address_id').customer_id.nunique())
d['n_pol_cust']=d.customer_id.map(pm.groupby('customer_id').policy_id.nunique()); d['n_lines_cust']=d.customer_id.map(pm.groupby('customer_id').l.nunique())
raw_src=pd.read_parquet(P+f'policies_{LN}.parquet',columns=['row_id','written_premium']); d['wp']=d.row_id.map(raw_src.set_index('row_id').written_premium)
d['wp_rate']=d.wp/d.exposure
tr=d.year<=2023; va=d.year==2024
def te_cols(df,cols,target,trmask,k=5,pw=200):
    out=pd.DataFrame(index=df.index); g=df[trmask]; mean=g[target].sum()/g.exposure.sum()
    for c in cols:
        enc=np.full(len(df),np.nan)
        for a,b in GroupKFold(k).split(g,groups=g.policy_id):
            s=g.iloc[a].groupby(c,observed=True).agg(y=(target,'sum'),e=('exposure','sum')); r=np.log((s.y+pw*mean)/(s.e+pw)/mean)
            enc[np.where(trmask)[0][b]]=g.iloc[b][c].map(r).fillna(0).values
        s=g.groupby(c,observed=True).agg(y=(target,'sum'),e=('exposure','sum')); r=np.log((s.y+pw*mean)/(s.e+pw)/mean)
        enc[~trmask.values]=df[~trmask][c].map(r).fillna(0).values; out['te_'+c]=enc
    return out
EXTRA=['t_cap','raw','t_cap100','wp','wp_rate']+HH
def run(name,target='t_cap',extra=[],params={},te=TE,seeds=[1]):
    t0=time.time(); X,cats=prep(d.drop(columns=EXTRA+te)); X=pd.concat([X,d[extra]],axis=1)
    if te: X=pd.concat([X,te_cols(d,te,target,tr)],axis=1)
    y=d[target].values; off=np.log(d.exposure.values)+np.log(y[tr].sum()/d.exposure[tr].sum()); mu=np.zeros(va.sum())
    for sd in seeds:
        prm={**BASE,**params,'seed':sd}
        ds=lgb.Dataset(X[tr],y[tr],init_score=off[tr],categorical_feature=cats); dv=lgb.Dataset(X[va],y[va],init_score=off[va],categorical_feature=cats,reference=ds)
        m=lgb.train(prm,ds,4000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
        mu+=np.exp(m.predict(X[va],raw_score=True)+off[va])/len(seeds)
    yr=d.raw[va].values; best=max((d2(yr,mu*s),s) for s in np.arange(0.6,1.5,0.025))
    print(f'{LN} {name:34s} it={m.best_iteration:4d} best D2={best[0]:.4f} @s={best[1]:.3f} {time.time()-t0:.0f}s',flush=True)
hh=HH if useHH else []
if os.environ.get('HIST'):
    HC=[c for c in d.columns if '_prev' in c or c.endswith('_anom')]
    run('v2 config (no hist)',extra=hh,params={}) if False else None
    EXTRA+= [c for c in HC]
    run('+ history+anom',extra=hh+HC)
    run('+ history only',extra=hh+[c for c in HC if '_prev' in c])
    run('+ hist, leaves15 min1000',extra=hh+HC,params=dict(num_leaves=15,min_data_in_leaf=1000,lambda_l2=30))
    sys.exit()
run('v2 config',extra=hh)
run('+ 3 seeds',extra=hh,seeds=[1,2,3])
run('+ written_premium',extra=hh+['wp_rate'])
run('cap 100k',target='t_cap100',extra=hh)
run('leaves63 min200',extra=hh,params=dict(num_leaves=63,min_data_in_leaf=200))
run('leaves15 min1000 l2=30',extra=hh,params=dict(num_leaves=15,min_data_in_leaf=1000,lambda_l2=30))
run('ff0.5 lr0.03',extra=hh,params=dict(feature_fraction=0.5,learning_rate=0.03))
run('TE pw=50',extra=hh) if False else None
