"""Overnight improvements, all scored on the same emerging GroupKFold(5) folds:
 (1) seed-averaged O and P  (2) pooled emerging weights 2/10/20  (3) freq-sev transfer prior T2 + offset model O2."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PL=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2)
PP=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2)
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=5)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=5)
def mk(df,cats=None,drop=()):
    X=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    if cats is None: cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    return X,cats
def align(df,X,cats,drop=()):
    Xt=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    return Xt[X.columns]
def te_map(df,c,pw=200):
    mean=df.target.sum()/df.exposure.sum(); s=df.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); return ((s.y+pw*mean)/(s.e+pw)/mean).apply(np.log)
SEEDS=[1,2,3,4,5]
for ln in (sys.argv[1].split(',') if len(sys.argv)>1 else ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet'); ye=em.target.values; ee=em.exposure.values
    folds=list(GroupKFold(5).split(em,groups=em.policy_id)); lvl_em=ye.sum()/ee.sum()
    # ---- (3) freq-sev transfer prior on developed (common features, region dropped)
    X,cats=mk(dv,drop=('region',)); Xe=align(em,X,cats,('region',)); Xte=align(te,X,cats,('region',))
    n=dv.n_claims.values; has=(n>0)&(dv.target.values>0); offF=np.log(dv.exposure.values)+np.log(n.sum()/dv.exposure.sum())
    mf=lgb.train(PF,lgb.Dataset(X,n,init_score=offF,categorical_feature=cats),400)
    ms=lgb.train(PS,lgb.Dataset(X[has],dv.target.values[has]/n[has],weight=n[has],categorical_feature=cats),120)
    T2_em=em.exposure.values*(n.sum()/dv.exposure.sum())*np.exp(mf.predict(Xe,raw_score=True))*ms.predict(Xe)
    T2_te=te.exposure.values*(n.sum()/dv.exposure.sum())*np.exp(mf.predict(Xte,raw_score=True))*ms.predict(Xte)
    l2=ye.sum()/T2_em.sum(); T2_em*=l2; T2_te*=l2
    print(f'{ln}: T2 (freq-sev transfer) D2 on emerging={d2(ye,T2_em):.4f} vs T={d2(ye,o["T"].values*ye.sum()/o["T"].sum()):.4f}',flush=True)
    # ---- local models: O (offset T), O2 (offset T2), seed-averaged
    XL,catsL=mk(em); XLt=align(te,XL,catsL)
    def local(offtr,offte,seeds):
        oof=np.zeros(len(em)); pt=np.zeros(len(te))
        for sd in seeds:
            its=[]
            for a,b in folds:
                m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL.iloc[a],ye[a],init_score=offtr[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=offtr[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
                oof[b]+=np.exp(m.predict(XL.iloc[b],raw_score=True)+offtr[b])/len(seeds); its.append(m.best_iteration)
            m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL,ye,init_score=offtr,categorical_feature=catsL),max(10,int(np.mean(its)*1.1)))
            pt+=np.exp(m.predict(XLt,raw_score=True)+offte)/len(seeds)
        return oof,pt
    O5,O5t=local(np.log(o['O'].values*0+o['T'].values*ye.sum()/o['T'].sum()),np.log(p['T'].values),SEEDS)
    O2,O2t=local(np.log(T2_em),np.log(T2_te),SEEDS)
    print(f'   O (1 seed)={d2(ye,o["O"].values):.4f}  O 5-seed={d2(ye,O5):.4f}  O2 (freq-sev prior, 5-seed)={d2(ye,O2):.4f}',flush=True)
    # ---- pooled with different weights, seed-averaged over 3
    for c in ['make_model','occupation']:
        if c in dv:
            mp=te_map(dv,c)
            for df in (dv,em,te): df['devte_'+c]=df[c].astype(str).str.strip().map(mp).fillna(0.0)
    dv['market']='dev'; em['market']='em'; te['market']='em'
    allx=pd.concat([dv,em],ignore_index=True); y=allx.target.values
    lvl={'dev':dv.target.sum()/dv.exposure.sum(),'em':lvl_em}; off=np.log(allx.exposure.values)+np.log(allx.market.map(lvl).values)
    XA,catsA=mk(allx); XAt=align(te,XA,catsA); is_em=(allx.market=='em').values; em_idx=np.where(is_em)[0]; dev_idx=np.where(~is_em)[0]
    best=None
    for EW in [2,10,20]:
        w=np.where(is_em,EW,1.0); oof=np.zeros(len(em)); pt=np.zeros(len(te)); seeds=[1,2,3]
        for sd in seeds:
            its=[]
            for a,b in folds:
                tr_idx=np.concatenate([dev_idx,em_idx[a]]); va=em_idx[b]
                m=lgb.train({**PP,'seed':sd},lgb.Dataset(XA.iloc[tr_idx],y[tr_idx],init_score=off[tr_idx],weight=w[tr_idx],categorical_feature=catsA),3000,valid_sets=[lgb.Dataset(XA.iloc[va],y[va],init_score=off[va],categorical_feature=catsA)],callbacks=[lgb.early_stopping(100,verbose=False)])
                oof[b]+=np.exp(m.predict(XA.iloc[va],raw_score=True)+off[va])/len(seeds); its.append(m.best_iteration)
            m=lgb.train({**PP,'seed':sd},lgb.Dataset(XA,y,init_score=off,weight=w,categorical_feature=catsA),max(20,int(np.mean(its)*1.1)))
            pt+=np.exp(m.predict(XAt,raw_score=True)+np.log(te.exposure.values)+np.log(lvl_em))/len(seeds)
        sc=d2(ye,oof); print(f'   P weight {EW} 3-seed: {sc:.4f} (P w5 1-seed was {d2(ye,o["P"].values):.4f})',flush=True)
        if best is None or sc>best[0]: best=(sc,EW,oof,pt)
    o['O5']=O5; o['O2']=O2; o['T2']=T2_em; o['P3']=best[2]; o.to_parquet(W2+f'oof_{ln}.parquet')
    p['O5']=O5t; p['O2']=O2t; p['T2']=T2_te; p['P3']=best[3]; p.to_parquet(W2+f'pred_{ln}.parquet')
    # stacking over all components
    from scipy.optimize import minimize
    names=['T','L','O','P','O5','O2','T2','P3']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(w): w=np.abs(w)/np.abs(w).sum(); return sum(wi*c for wi,c in zip(w,comps))
    r=minimize(lambda w: tweedie_dev(ye,blend(w)).sum(),[1/8]*8,method='Nelder-Mead',options={'maxiter':4000}); w=np.abs(r.x)/np.abs(r.x).sum()
    print(f'   STACK {dict(zip(names,np.round(w,2)))} OOF D2={d2(ye,blend(w)):.4f}  (v3 blend was {json.load(open(W2+"stack_p2_v3.json"))[ln]["d2_blend"]:.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v4.json')) if __import__('os').path.exists(W2+'stack_p2_v4.json') else {}
    st[ln]=dict(names=names,w=w.tolist(),d2=float(d2(ye,blend(w)))); json.dump(st,open(W2+'stack_p2_v4.json','w'),indent=1)
print('done')
