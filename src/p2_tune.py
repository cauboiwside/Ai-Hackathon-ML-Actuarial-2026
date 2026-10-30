"""Quick local-model experiments on the offset model O2 (prior T2): history features, hyperparameter grid, Tweedie power."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
BASE=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2,seed=1)
for ln in ['motor','buildings','contents']:
    em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); o=pd.read_parquet(W2+f'oof_{ln}.parquet'); ye=em.target.values; off=np.log(o['T2'].values)
    j=pd.read_parquet(W2+f'claimrows_em_{ln}.parquet').merge(em[['row_id','policy_id','year']],on='row_id')
    h=j[j.year==2023].groupby('policy_id').agg(prev_n=('claim_id','size'),prev_c=('cost_c','sum'))
    em['prev_n']=np.where(em.year==2024,em.policy_id.map(h.prev_n).fillna(0),np.nan); em['prev_c']=np.where(em.year==2024,em.policy_id.map(h.prev_c).fillna(0),np.nan)
    folds=list(GroupKFold(5).split(em,groups=em.policy_id))
    def run(params,extra=()):
        X=em.drop(columns=[c for c in ID+EXCL+['prev_n','prev_c'] if c in em and c not in extra]); cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
        for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
        oof=np.zeros(len(em))
        for a,b in folds:
            m=lgb.train(params,lgb.Dataset(X.iloc[a],ye[a],init_score=off[a],categorical_feature=cats),2000,valid_sets=[lgb.Dataset(X.iloc[b],ye[b],init_score=off[b],categorical_feature=cats)],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[b]=np.exp(m.predict(X.iloc[b],raw_score=True)+off[b])
        return d2(ye,oof)
    base=run(BASE); res={'base':base,'+history':run(BASE,('prev_n','prev_c'))}
    for nl,md,l2 in [(4,300,30),(16,300,30),(8,100,30),(8,1000,30),(8,300,5),(8,300,100),(31,500,10)]: res[f'leaves{nl}_min{md}_l2{l2}']=run({**BASE,'num_leaves':nl,'min_data_in_leaf':md,'lambda_l2':l2})
    for p in [1.3,1.7]: res[f'power{p}']=run({**BASE,'tweedie_variance_power':p})
    best=max(res,key=res.get)
    print(f'{ln}: '+' | '.join(f'{k} {v:.4f}' for k,v in res.items())+f'  => best {best} (+{res[best]-base:.4f})',flush=True)
print('done')
