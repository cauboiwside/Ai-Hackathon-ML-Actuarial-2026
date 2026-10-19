"""Validation: train on 2021-2023, score 2024 (out-of-time). Checks level + ranking."""
import sys; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from common import *
PRM=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,
         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=2,seed=1,cat_smooth=50,min_data_per_group=200,cat_l2=10)
for ln in ['motor','buildings','contents']:
    d=pd.read_parquet(W+f'mf_train_{ln}.parquet')
    X,cats=prep(d)
    tr=d.year<=2023; va=d.year==2024
    ds=lgb.Dataset(X[tr],d.target[tr],init_score=np.log(d.exposure[tr]),categorical_feature=cats,free_raw_data=False)
    dv=lgb.Dataset(X[va],d.target[va],init_score=np.log(d.exposure[va]),categorical_feature=cats,reference=ds)
    m=lgb.train(PRM,ds,2000,valid_sets=[dv],callbacks=[lgb.early_stopping(100,verbose=False)])
    mu=m.predict(X[va],raw_score=True); mu=np.exp(mu)*d.exposure[va].values
    y=d.target[va].values
    flat=np.full_like(y,y.sum()/d.exposure[va].sum())*d.exposure[va].values
    print(f'{ln}: iters={m.best_iteration} D2_model={d2(y,mu):.4f} D2_expo_only={d2(y,flat):.4f} actual/pred={y.sum()/mu.sum():.3f}')
    imp=pd.Series(m.feature_importance('gain'),index=X.columns).sort_values(ascending=False)
    print('  top:',', '.join(f'{k}:{v/imp.sum():.2f}' for k,v in imp.head(15).items()))
    print('  zero-gain:',list(imp[imp==0].index))
