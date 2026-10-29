"""Part 2 models per line:
 T  transfer: LightGBM Tweedie on developed market, common features (region dropped: codes differ) -> predicts emerging rows
 L  local:    LightGBM Tweedie on emerging data only (2 yrs), strong regularisation, GroupKFold by policy
 O  offset:   local model with init_score = log(T prediction): learns deviations from the developed-market prior (credibility)
Outputs OOF for L and O on emerging train, T/L/O predictions on emerging test, all at the emerging pooled level."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PT=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=1)
PL=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2,seed=1)
def prep(df,cats=None,drop_region=False):
    X=df.drop(columns=[c for c in ID+EXCL if c in df.columns]+(['region'] if drop_region else []))
    if cats is None: cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    return X,cats
def align(Xt,X,cats):
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    return Xt[X.columns]
res={}
for ln in (sys.argv[1:] or ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    # ---- T: transfer model on developed
    X,cats=prep(dv,drop_region=True); y=dv.target.values; off=np.log(dv.exposure.values)+np.log(y.sum()/dv.exposure.sum())
    mT=lgb.train(PT,lgb.Dataset(X,y,init_score=off,categorical_feature=cats),250)
    Xe=align(prep(em,cats,drop_region=True)[0],X,cats); Xte=align(prep(te,cats,drop_region=True)[0],X,cats)
    base_dev=y.sum()/dv.exposure.sum()
    T_em=em.exposure.values*base_dev*np.exp(mT.predict(Xe,raw_score=True)); T_te=te.exposure.values*base_dev*np.exp(mT.predict(Xte,raw_score=True))
    ye=em.target.values; lvl=ye.sum()/T_em.sum()           # emerging/developed level ratio at matched risk mix
    print(f'{ln}: transfer level ratio emerging/developed = {lvl:.3f} | T OOF-free D2 on emerging (level-matched) = {d2(ye,T_em*lvl):.4f}',flush=True)
    # ---- L and O on emerging with GroupKFold
    XL,catsL=prep(em); XLt=align(prep(te,catsL)[0],XL,catsL)
    folds=list(GroupKFold(5).split(XL,ye,em.policy_id)); offE=np.log(em.exposure.values)+np.log(ye.sum()/em.exposure.sum())
    offO=np.log(T_em*lvl)
    oofL=np.zeros(len(em)); oofO=np.zeros(len(em)); itL=[]; itO=[]
    for a,b in folds:
        m=lgb.train(PL,lgb.Dataset(XL.iloc[a],ye[a],init_score=offE[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=offE[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
        oofL[b]=np.exp(m.predict(XL.iloc[b],raw_score=True)+offE[b]); itL.append(m.best_iteration)
        m=lgb.train(PL,lgb.Dataset(XL.iloc[a],ye[a],init_score=offO[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=offO[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
        oofO[b]=np.exp(m.predict(XL.iloc[b],raw_score=True)+offO[b]); itO.append(m.best_iteration)
    Tm=T_em*lvl
    print(f'   L iters {itL} OOF D2={d2(ye,oofL):.4f} | O iters {itO} OOF D2={d2(ye,oofO):.4f} | T={d2(ye,Tm):.4f} | blends: T+L {d2(ye,(Tm+oofL)/2):.4f}  T+O {d2(ye,(Tm+oofO)/2):.4f}  L+O {d2(ye,(oofL+oofO)/2):.4f}  all3 {d2(ye,(Tm+oofL+oofO)/3):.4f}',flush=True)
    mL=lgb.train(PL,lgb.Dataset(XL,ye,init_score=offE,categorical_feature=catsL),max(10,int(np.mean(itL)*1.1)))
    mO=lgb.train(PL,lgb.Dataset(XL,ye,init_score=offO,categorical_feature=catsL),max(10,int(np.mean(itO)*1.1)))
    L_te=np.exp(mL.predict(XLt,raw_score=True)+np.log(te.exposure.values)+np.log(ye.sum()/em.exposure.sum()))
    O_te=np.exp(mO.predict(XLt,raw_score=True)+np.log(T_te*lvl))
    pd.DataFrame({'row_id':em.row_id,'y':ye,'exposure':em.exposure,'T':Tm,'L':oofL,'O':oofO,'year':em.year,'new_business':em.new_business}).to_parquet(W2+f'oof_{ln}.parquet')
    pd.DataFrame({'row_id':te.row_id,'exposure':te.exposure,'T':T_te*lvl,'L':L_te,'O':O_te,'new_business':te.new_business}).to_parquet(W2+f'pred_{ln}.parquet')
    imp=pd.Series(mO.feature_importance('gain'),index=XL.columns); print('   O top features:',', '.join(f'{k}:{v/imp.sum():.2f}' for k,v in imp.sort_values(ascending=False).head(8).items()))
    res[ln]=dict(level_ratio=float(lvl),itL=itL,itO=itO,d2_T=d2(ye,Tm),d2_L=d2(ye,oofL),d2_O=d2(ye,oofO))
json.dump(res,open(W2+'p2_results.json','w'),indent=1,default=float); print('done')
