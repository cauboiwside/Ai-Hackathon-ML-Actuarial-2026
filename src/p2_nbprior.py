"""NB prior: freq-sev transfer model trained on developed data with new-business rows weighted x8 (so the fit is ~57% NB),
plus local offset model on it. Evaluated on all emerging OOF rows and on NB rows separately."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=21)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=21)
PL=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2)
NBW=8.0
def mk(df,cats=None,drop=()):
    X=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    if cats is None: cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    return X,cats
def align(df,X,cats,drop=()):
    Xt=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    return Xt[X.columns]
for ln in (sys.argv[1].split(',') if len(sys.argv)>1 else ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    X,cats=mk(dv,drop=('region',)); Xe=align(em,X,cats,('region',)); Xte=align(te,X,cats,('region',))
    w=np.where(dv.new_business.values,NBW,1.0); n=dv.n_claims.values; y=dv.target.values; has=(n>0)&(y>0)
    br=(n*w).sum()/(dv.exposure.values*w).sum(); offF=np.log(dv.exposure.values)+np.log(br)
    mf=lgb.train(PF,lgb.Dataset(X,n,init_score=offF,weight=w,categorical_feature=cats),400)
    ms=lgb.train(PS,lgb.Dataset(X[has],y[has]/n[has],weight=n[has]*w[has],categorical_feature=cats),120)
    T_em=em.exposure.values*br*np.exp(mf.predict(Xe,raw_score=True))*ms.predict(Xe); T_te=te.exposure.values*br*np.exp(mf.predict(Xte,raw_score=True))*ms.predict(Xte)
    ye=em.target.values; nb=em.new_business.values; l=ye.sum()/T_em.sum(); T_em*=l; T_te*=l
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    def sc(v): v=v*ye.sum()/v.sum(); return f'all {d2(ye,v):.4f} NB {d2(ye[nb],v[nb]*ye[nb].sum()/v[nb].sum()):.4f} REN {d2(ye[~nb],v[~nb]*ye[~nb].sum()/v[~nb].sum()):.4f}'
    print(f'{ln}: TNB (NB-weighted freq-sev prior): {sc(T_em)} | T2: {sc(o["T2"].values)} | T3: {sc(o["T3"].values)}',flush=True)
    XL,catsL=mk(em); XLt=align(te,XL,catsL); folds=list(GroupKFold(5).split(em,groups=em.policy_id)); oof=np.zeros(len(em)); pt=np.zeros(len(te)); lo=np.log(T_em); lt=np.log(T_te)
    for sd in [1,2,3]:
        its=[]
        for a,b in folds:
            m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL.iloc[a],ye[a],init_score=lo[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=lo[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[b]+=np.exp(m.predict(XL.iloc[b],raw_score=True)+lo[b])/3; its.append(m.best_iteration)
        m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL,ye,init_score=lo,categorical_feature=catsL),max(10,int(np.mean(its)*1.1))); pt+=np.exp(m.predict(XLt,raw_score=True)+lt)/3
    print(f'   ONB (offset on TNB): {sc(oof)} | O2: {sc(o["O2"].values)}',flush=True)
    o['TNB']=T_em; o['ONB']=oof; p['TNB']=T_te; p['ONB']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
print('done')
