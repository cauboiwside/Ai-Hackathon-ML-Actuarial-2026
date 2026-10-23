"""Second in-house model: frequency (Poisson GBM on capped claim count, log-exposure offset) x severity
(Gamma GBM on capped cost per claim, claim-level, weighted by count). Same features/folds as v2.
Produces OOF pure premium and test pure premium at the v2 pooled level (level factors applied in blend)."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
P='/home/claude/data_raw/public/'
PF=dict(objective='poisson',learning_rate=0.03,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=7)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=7)
TE={'motor':['make_model','occupation','region'],'buildings':['occupation','region'],'contents':[]}
HHC={'motor':True,'buildings':False,'contents':False}; HH=['n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']
cl=pd.read_parquet(W+'claims_level.parquet')
pm=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['row_id','policy_id','customer_id','address_id']).assign(l=f[-1]) for f in ['policies_motor','policies_buildings','policies_contents','test_motor','test_buildings','test_contents']])
def hh(df):
    df['n_pol_addr']=df.address_id.map(pm.groupby('address_id').policy_id.nunique()); df['n_cust_addr']=df.address_id.map(pm.groupby('address_id').customer_id.nunique())
    df['n_pol_cust']=df.customer_id.map(pm.groupby('customer_id').policy_id.nunique()); df['n_lines_cust']=df.customer_id.map(pm.groupby('customer_id').l.nunique()); return df
def te_fit(g,c,pw=200):
    mean=g.target.sum()/g.exposure.sum(); s=g.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); return ((s.y+pw*mean)/(s.e+pw)/mean).apply(np.log)
out={}; preds=[]
for ln in ['motor','buildings','contents']:
    d=hh(pd.read_parquet(W+f'mf_train_{ln}.parquet')); te=hh(pd.read_parquet(W+f'mf_test_{ln}.parquet'))
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(cl,on='claim_id'); j=j[j.cost_c>0]
    d['target']=d.row_id.map(j.groupby('row_id').cost_c.sum()).fillna(0.0); d['n']=d.row_id.map(j.groupby('row_id').size()).fillna(0).astype(int)
    dropc=TE[ln]+([] if HHC[ln] else HH)
    X,cats=prep(d.drop(columns=dropc+['n'])); Xt,_=prep(te.drop(columns=dropc),cats)
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str),categories=X[c].cat.categories)
    folds=list(GroupKFold(4).split(X,d.target,d.policy_id))
    for c in TE[ln]:
        X['te_'+c]=np.nan
        for a,b in folds: X.iloc[b,X.columns.get_loc('te_'+c)]=d.iloc[b][c].map(te_fit(d.iloc[a],c)).fillna(0).values
        Xt['te_'+c]=te[c].map(te_fit(d,c)).fillna(0).values
    Xt=Xt[X.columns]
    n=d.n.values; off=np.log(d.exposure.values); y=d.target.values; oof=np.zeros(len(d)); itf=[]; its=[]
    has=n>0; sev_y=y[has]/n[has]
    for a,b in folds:
        mf=lgb.train(PF,lgb.Dataset(X.iloc[a],n[a],init_score=off[a],categorical_feature=cats),4000,valid_sets=[lgb.Dataset(X.iloc[b],n[b],init_score=off[b],categorical_feature=cats)],callbacks=[lgb.early_stopping(150,verbose=False)])
        ah=a[has[a]]; bh=b[has[b]]
        ms=lgb.train(PS,lgb.Dataset(X.iloc[ah],y[ah]/n[ah],weight=n[ah],categorical_feature=cats),4000,valid_sets=[lgb.Dataset(X.iloc[bh],y[bh]/n[bh],weight=n[bh],categorical_feature=cats)],callbacks=[lgb.early_stopping(150,verbose=False)])
        oof[b]=np.exp(mf.predict(X.iloc[b],raw_score=True)+off[b])*ms.predict(X.iloc[b]); itf.append(mf.best_iteration); its.append(ms.best_iteration)
    print(f'{ln}: freq iters {itf} sev iters {its} | OOF D2 freq-sev={d2(y,oof):.4f} | v2 OOF D2={d2(y,pd.read_parquet(W+f"oof_v2_{ln}.parquet").oof.values):.4f}',flush=True)
    o2=pd.read_parquet(W+f'oof_v2_{ln}.parquet'); assert (o2.row_id.values==d.row_id.values).all()
    for wfs in [0.3,0.5]: print(f'   blend v2/fs {1-wfs:.1f}/{wfs:.1f}: OOF D2={d2(y,(1-wfs)*o2.oof.values+wfs*oof):.4f}')
    mf=lgb.train(PF,lgb.Dataset(X,n,init_score=off,categorical_feature=cats),int(np.mean(itf)*1.1))
    ms=lgb.train(PS,lgb.Dataset(X[has],sev_y,weight=n[has],categorical_feature=cats),int(np.mean(its)*1.1))
    pt=te.exposure.values*np.exp(mf.predict(Xt,raw_score=True))*ms.predict(Xt)
    preds.append(pd.DataFrame({'row_id':te.row_id,'prediction':pt}))
    pd.DataFrame({'row_id':d.row_id,'y':y,'oof_fs':oof,'exposure':d.exposure}).to_parquet(W+f'oof_fs_{ln}.parquet')
pd.concat(preds).to_csv(W+'pred_fs.csv',index=False); print('done')
