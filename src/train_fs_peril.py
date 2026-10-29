"""Per-peril frequency x severity: for each line, for each peril, Poisson frequency GBM (all rows, log-exposure offset)
x Gamma severity GBM (claims of that peril). Expected cost = sum over perils. Same features/folds as v2/fs."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
P='/home/claude/data_raw/public/'
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=11)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=11)
TE={'motor':['make_model','occupation','region'],'buildings':['occupation','region'],'contents':[]}
HHC={'motor':True,'buildings':False,'contents':False}; HH=['n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']
cl=pd.read_parquet(W+'claims_level.parquet'); cp=pd.read_parquet(W+'claims_prepped.parquet',columns=['claim_id','peril'])
pass  # claims_level already has peril
pm=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['row_id','policy_id','customer_id','address_id']).assign(l=f[-1]) for f in ['policies_motor','policies_buildings','policies_contents','test_motor','test_buildings','test_contents']])
def hh(df):
    df['n_pol_addr']=df.address_id.map(pm.groupby('address_id').policy_id.nunique()); df['n_cust_addr']=df.address_id.map(pm.groupby('address_id').customer_id.nunique())
    df['n_pol_cust']=df.customer_id.map(pm.groupby('customer_id').policy_id.nunique()); df['n_lines_cust']=df.customer_id.map(pm.groupby('customer_id').l.nunique()); return df
def te_fit(g,c,pw=200):
    mean=g.target.sum()/g.exposure.sum(); s=g.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); return ((s.y+pw*mean)/(s.e+pw)/mean).apply(np.log)
LINES=sys.argv[1:] or ['motor','buildings','contents']
for ln in LINES:
    d=hh(pd.read_parquet(W+f'mf_train_{ln}.parquet')); te=hh(pd.read_parquet(W+f'mf_test_{ln}.parquet'))
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(cl,on='claim_id'); j=j[j.cost_c>0]
    d['target']=d.row_id.map(j.groupby('row_id').cost_c.sum()).fillna(0.0)
    dropc=TE[ln]+([] if HHC[ln] else HH)
    X,cats=prep(d.drop(columns=dropc)); Xt,_=prep(te.drop(columns=dropc),cats)
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str),categories=X[c].cat.categories)
    import os
    folds=list(GroupKFold(int(os.environ.get('FOLDS','4'))).split(X,d.target,d.policy_id)); PAT=int(os.environ.get('PATIENCE','100'))
    for c in TE[ln]:
        X['te_'+c]=np.nan
        for a,b in folds: X.iloc[b,X.columns.get_loc('te_'+c)]=d.iloc[b][c].map(te_fit(d.iloc[a],c)).fillna(0).values
        Xt['te_'+c]=te[c].map(te_fit(d,c)).fillna(0).values
    Xt=Xt[X.columns]; off=np.log(d.exposure.values); y=d.target.values
    oof=np.zeros(len(d)); pt=np.zeros(len(te))
    for peril,jp in j.groupby('peril'):
        n=d.row_id.map(jp.groupby('row_id').size()).fillna(0).astype(int).values; cst=d.row_id.map(jp.groupby('row_id').cost_c.sum()).fillna(0).values
        has=n>0; itf=[]; its=[]
        offp=off+np.log(n.sum()/d.exposure.sum())   # start frequency at the peril's base rate
        for a,b in folds:
            mf=lgb.train(PF,lgb.Dataset(X.iloc[a],n[a],init_score=offp[a],categorical_feature=cats),3000,valid_sets=[lgb.Dataset(X.iloc[b],n[b],init_score=offp[b],categorical_feature=cats)],callbacks=[lgb.early_stopping(PAT,verbose=False)])
            ah=a[has[a]]; bh=b[has[b]]
            ms=lgb.train(PS,lgb.Dataset(X.iloc[ah],cst[ah]/n[ah],weight=n[ah],categorical_feature=cats),3000,valid_sets=[lgb.Dataset(X.iloc[bh],cst[bh]/n[bh],weight=n[bh],categorical_feature=cats)],callbacks=[lgb.early_stopping(PAT,verbose=False)])
            oof[b]+=np.exp(mf.predict(X.iloc[b],raw_score=True)+offp[b])*ms.predict(X.iloc[b]); itf.append(mf.best_iteration); its.append(ms.best_iteration)
        mf=lgb.train(PF,lgb.Dataset(X,n,init_score=offp,categorical_feature=cats),max(20,int(np.mean(itf)*1.1)))
        ms=lgb.train(PS,lgb.Dataset(X[has],cst[has]/n[has],weight=n[has],categorical_feature=cats),max(20,int(np.mean(its)*1.1)))
        pt+=te.exposure.values*(n.sum()/d.exposure.sum())*np.exp(mf.predict(Xt,raw_score=True))*ms.predict(Xt)
        print(f'  {ln}/{peril}: n_claims={int(n.sum())} freq it {itf} sev it {its}',flush=True)
    o2=pd.read_parquet(W+f'oof_v2_{ln}.parquet'); of=pd.read_parquet(W+f'oof_fs_{ln}.parquet')
    print(f'{ln}: OOF D2 per-peril={d2(y,oof):.4f} | fs={d2(y,of.oof_fs.values):.4f} | v2={d2(y,o2.oof.values):.4f} | blend(v2,fs,peril)/3={d2(y,(o2.oof.values+of.oof_fs.values+oof)/3):.4f} | fs+peril 50/50={d2(y,(of.oof_fs.values+oof)/2):.4f}',flush=True)
    pd.DataFrame({'row_id':d.row_id,'y':y,'oof_pp':oof}).to_parquet(W+f'oof_fsp_{ln}.parquet')
    pd.DataFrame({'row_id':te.row_id,'prediction':pt}).to_csv(W+f'pred_fsp_{ln}.csv',index=False)
print('done')
