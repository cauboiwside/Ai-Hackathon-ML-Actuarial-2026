"""Pooled two-market frequency x severity model (PF2): Poisson frequency and Gamma severity each trained on
developed+emerging with market flag, emerging weight 2, developed TE priors. Emerging GroupKFold(5) OOF; 3 seeds."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2)
EW=2.0
def te_map(df,c,pw=200):
    mean=df.target.sum()/df.exposure.sum(); s=df.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); return ((s.y+pw*mean)/(s.e+pw)/mean).apply(np.log)
for ln in (sys.argv[1].split(',') if len(sys.argv)>1 else ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    for c in ['make_model','occupation']:
        if c in dv:
            mp=te_map(dv,c)
            for df in (dv,em,te): df['devte_'+c]=df[c].astype(str).str.strip().map(mp).fillna(0.0)
    dv['market']='dev'; em['market']='em'; te['market']='em'
    allx=pd.concat([dv,em],ignore_index=True); y=allx.target.values; e=allx.exposure.values
    n=allx.n_claims.values.astype(int); n=np.where(y>0,np.maximum(n,1),0)      # count only positive-cost claims
    fr={'dev':n[~(allx.market=='em').values].sum()/dv.exposure.sum(),'em':n[(allx.market=='em').values].sum()/em.exposure.sum()}
    offF=np.log(e)+np.log(allx.market.map(fr).values); has=n>0
    X=allx.drop(columns=[c for c in ID+EXCL if c in allx]); cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    Xt=te.drop(columns=[c for c in ID+EXCL if c in te])
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    Xt=Xt[X.columns]
    is_em=(allx.market=='em').values; em_idx=np.where(is_em)[0]; dev_idx=np.where(~is_em)[0]; w=np.where(is_em,EW,1.0)
    oof=np.zeros(len(em)); pt=np.zeros(len(te)); seeds=[1,2,3]
    for sd in seeds:
        itf=[]; its=[]
        for a,b in GroupKFold(5).split(em,groups=em.policy_id):
            tr=np.concatenate([dev_idx,em_idx[a]]); va=em_idx[b]
            mf=lgb.train({**PF,'seed':sd},lgb.Dataset(X.iloc[tr],n[tr],init_score=offF[tr],weight=w[tr],categorical_feature=cats),3000,valid_sets=[lgb.Dataset(X.iloc[va],n[va],init_score=offF[va],categorical_feature=cats)],callbacks=[lgb.early_stopping(100,verbose=False)])
            trh=tr[has[tr]]; vah=va[has[va]]
            ms=lgb.train({**PS,'seed':sd},lgb.Dataset(X.iloc[trh],y[trh]/n[trh],weight=n[trh]*w[trh],categorical_feature=cats),3000,valid_sets=[lgb.Dataset(X.iloc[vah],y[vah]/n[vah],weight=n[vah],categorical_feature=cats)],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[b]+=np.exp(mf.predict(X.iloc[va],raw_score=True)+offF[va])*ms.predict(X.iloc[va])/len(seeds); itf.append(mf.best_iteration); its.append(ms.best_iteration)
        mf=lgb.train({**PF,'seed':sd},lgb.Dataset(X,n,init_score=offF,weight=w,categorical_feature=cats),max(20,int(np.mean(itf)*1.1)))
        ms=lgb.train({**PS,'seed':sd},lgb.Dataset(X[has],y[has]/n[has],weight=n[has]*w[has],categorical_feature=cats),max(20,int(np.mean(its)*1.1)))
        pt+=te.exposure.values*fr['em']*np.exp(mf.predict(Xt,raw_score=True))*ms.predict(Xt)/len(seeds)
        print(f'  {ln} seed {sd}: freq it {itf} sev it {its}',flush=True)
    ye=em.target.values; o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    print(f'{ln}: PF2 pooled freq-sev OOF D2={d2(ye,oof*ye.sum()/oof.sum()):.4f} | P3={d2(ye,o["P3"].values):.4f} O2={d2(ye,o["O2"].values):.4f}',flush=True)
    o['PF2']=oof; p['PF2']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
    names=['T','L','O','P','O5','O2','T2','P3','T3','O3','PF2']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(wv): wv=np.abs(wv)/np.abs(wv).sum(); return sum(wi*c for wi,c in zip(wv,comps))
    r=minimize(lambda wv: tweedie_dev(ye,blend(wv)).sum(),[0.1]*11,method='Nelder-Mead',options={'maxiter':8000}); wv=np.abs(r.x)/np.abs(r.x).sum()
    v5=json.load(open(W2+'stack_p2_v5.json'))[ln]['d2']
    print(f'   STACK v6 {dict(zip(names,np.round(wv,2)))} OOF D2={d2(ye,blend(wv)):.4f} (v5 {v5:.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v6.json')) if os.path.exists(W2+'stack_p2_v6.json') else {}
    st[ln]=dict(names=names,w=wv.tolist(),d2=float(d2(ye,blend(wv)))); json.dump(st,open(W2+'stack_p2_v6.json','w'),indent=1)
print('done')
