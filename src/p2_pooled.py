"""Pooled-market model P: train on developed + emerging together (common features), with market flag and
sample weight on emerging rows, Tweedie 1.5. Validation: GroupKFold on emerging rows only (developed rows always in train).
Also adds developed-market target encodings of make_model/occupation as numeric priors."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PP=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=3)
EW=float(sys.argv[2]) if len(sys.argv)>2 else 5.0     # weight on emerging rows
def te_map(df,c,pw=200):
    mean=df.target.sum()/df.exposure.sum(); s=df.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); return ((s.y+pw*mean)/(s.e+pw)/mean).apply(np.log)
for ln in (sys.argv[1].split(',') if len(sys.argv)>1 else ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    for c in ['make_model','occupation']:
        if c in dv: 
            mp=te_map(dv,c)
            for df in (dv,em,te): df['devte_'+c]=df[c].astype(str).str.strip().map(mp).fillna(0.0)
    dv['market']='dev'; em['market']='em'; te['market']='em'
    # region: keep as categorical; developed codes never appear in emerging rows, trees split on market first
    allx=pd.concat([dv,em],ignore_index=True); y=allx.target.values; e=allx.exposure.values
    lvl={'dev':dv.target.sum()/dv.exposure.sum(),'em':em.target.sum()/em.exposure.sum()}
    off=np.log(e)+np.log(allx.market.map(lvl).values)      # each market starts at its own level
    X=allx.drop(columns=[c for c in ID+EXCL if c in allx]); cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    Xt=te.drop(columns=[c for c in ID+EXCL if c in te])
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    Xt=Xt[X.columns]
    w=np.where(allx.market=='em',EW,1.0); is_em=(allx.market=='em').values; em_idx=np.where(is_em)[0]
    oof=np.zeros(len(em)); its=[]
    for a,b in GroupKFold(5).split(em,groups=em.policy_id):
        tr_idx=np.concatenate([np.where(~is_em)[0],em_idx[a]]); va_idx=em_idx[b]
        m=lgb.train(PP,lgb.Dataset(X.iloc[tr_idx],y[tr_idx],init_score=off[tr_idx],weight=w[tr_idx],categorical_feature=cats),3000,
                    valid_sets=[lgb.Dataset(X.iloc[va_idx],y[va_idx],init_score=off[va_idx],categorical_feature=cats)],callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[b]=np.exp(m.predict(X.iloc[va_idx],raw_score=True)+off[va_idx]); its.append(m.best_iteration)
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); ye=em.target.values
    prev=[o[c].values*ye.sum()/o[c].sum() for c in ['T','L','O']]; P=oof*ye.sum()/oof.sum()
    print(f'{ln} (em weight {EW}): iters {its} | P OOF D2={d2(ye,P):.4f} | T={d2(ye,prev[0]):.4f} O={d2(ye,prev[2]):.4f} | P+O {d2(ye,(P+prev[2])/2):.4f} P+T+L {d2(ye,(P+prev[0]+prev[1])/3):.4f} P+T+L+O {d2(ye,(P+prev[0]+prev[1]+prev[2])/4):.4f}',flush=True)
    m=lgb.train(PP,lgb.Dataset(X,y,init_score=off,weight=w,categorical_feature=cats),max(20,int(np.mean(its)*1.1)))
    pt=np.exp(m.predict(Xt,raw_score=True)+np.log(te.exposure.values)+np.log(lvl['em']))
    o['P']=oof; o.to_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet'); p['P']=pt; p.to_parquet(W2+f'pred_{ln}.parquet')
    imp=pd.Series(m.feature_importance('gain'),index=X.columns); print('   top:',', '.join(f'{k}:{v/imp.sum():.2f}' for k,v in imp.sort_values(ascending=False).head(10).items()),flush=True)
print('done')
