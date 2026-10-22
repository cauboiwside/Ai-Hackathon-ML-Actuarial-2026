"""v2: capped raw cost target (aggregate-level adjustments applied after), OOF target encoding of
high-cardinality categoricals, household count features, LightGBM Tweedie 1.5 with exposure offset,
4-fold CV grouped by policy, then calibration curve fitted on OOF and aggregate level factors."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import *
P='/home/claude/data_raw/public/'
PRM=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=31,min_data_in_leaf=500,
         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,
         cat_l2=10,verbose=-1,num_threads=2,seed=1)
# per-line config chosen on 2024 holdout vs raw actuals (docs/decision_log.md #13-14)
TE={'motor':['make_model','occupation','region'],'buildings':['occupation','region'],'contents':[]}
HHC={'motor':True,'buildings':False,'contents':False}
HH=['n_pol_addr','n_cust_addr','n_pol_cust','n_lines_cust']
lp=json.load(open(W+'level_params.json')); cl=pd.read_parquet(W+'claims_level.parquet')
pm=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['row_id','policy_id','customer_id','address_id']).assign(l=f[-1])
              for f in ['policies_motor','policies_buildings','policies_contents','test_motor','test_buildings','test_contents']])
def hh(df):
    df['n_pol_addr']=df.address_id.map(pm.groupby('address_id').policy_id.nunique())
    df['n_cust_addr']=df.address_id.map(pm.groupby('address_id').customer_id.nunique())
    df['n_pol_cust']=df.customer_id.map(pm.groupby('customer_id').policy_id.nunique())
    df['n_lines_cust']=df.customer_id.map(pm.groupby('customer_id').l.nunique()); return df
def te_fit(g,c,prior_w=200):
    mean=g.target.sum()/g.exposure.sum(); s=g.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum'))
    return ((s.y+prior_w*mean)/(s.e+prior_w)/mean).apply(np.log), mean
out={}; preds=[]
for ln in ['motor','buildings','contents']:
    d=hh(pd.read_parquet(W+f'mf_train_{ln}.parquet')); te=hh(pd.read_parquet(W+f'mf_test_{ln}.parquet'))
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(cl,on='claim_id')
    d['target']=d.row_id.map(j.groupby('row_id').cost_c.sum()).fillna(0.0)   # capped raw cost, no per-claim weights
    dropc=TE[ln]+([] if HHC[ln] else HH)
    X,cats=prep(d.drop(columns=dropc)); Xt,_=prep(te.drop(columns=dropc),cats)
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str),categories=X[c].cat.categories)
    y=d.target.values; off=np.log(d.exposure.values)+np.log(y.sum()/d.exposure.sum())
    folds=list(GroupKFold(4).split(X,y,d.policy_id)); oof=np.zeros(len(d)); its=[]
    # OOF target encoding: encoding for fold b computed from the other folds only
    for c in TE[ln]:
        X['te_'+c]=np.nan
        for a,b in folds:
            r,mean=te_fit(d.iloc[a],c); X.iloc[b,X.columns.get_loc('te_'+c)]=d.iloc[b][c].map(r).fillna(0).values
        r,mean=te_fit(d,c); Xt['te_'+c]=te[c].map(r).fillna(0).values
    Xt=Xt[X.columns]
    for a,b in folds:
        ds=lgb.Dataset(X.iloc[a],y[a],init_score=off[a],categorical_feature=cats)
        dv=lgb.Dataset(X.iloc[b],y[b],init_score=off[b],categorical_feature=cats,reference=ds)
        m=lgb.train(PRM,ds,4000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
        oof[b]=np.exp(m.predict(X.iloc[b],raw_score=True)+off[b]); its.append(m.best_iteration)
    D2=d2(y,oof); print(f'{ln}: iters={its} OOF D2 (capped raw)={D2:.4f}',flush=True)
    # calibration curve on OOF rate
    lr=np.log(oof/d.exposure.values); e=d.exposure.values
    r=minimize(lambda p: tweedie_dev(y,e*np.exp(p[0]+p[1]*lr)).sum(),[0,1],method='Nelder-Mead'); c0,a=r.x
    print(f'   calib a={a:.3f} c={np.exp(c0):.3f} D2 after={d2(y,e*np.exp(c0+a*lr)):.4f} A/E={y.sum()/(e*np.exp(c0+a*lr)).sum():.3f}')
    m=lgb.train(PRM,lgb.Dataset(X,y,init_score=off,categorical_feature=cats),int(np.mean(its)*1.1)); m.save_model(W+f'lgb_v2_{ln}.txt')
    rate=np.exp(m.predict(Xt,raw_score=True)+np.log(y.sum()/d.exposure.sum()))
    # aggregate level to 2025 ultimate: IBNR+dev on 2024 is implicit in pooled level, so apply
    # trend from mid-point of training (2022.5) to 2025 and the large-loss load
    lvl=(1+lp['trend'][ln])**2.5*lp['ll'][ln]
    pt_unbiased=te.exposure.values*rate*lvl
    pt_metric=te.exposure.values*np.exp(c0+a*np.log(rate))*lvl
    print(f'   level factor={lvl:.3f} test pp unbiased={pt_unbiased.sum()/te.exposure.sum():.1f} metric-optimal={pt_metric.sum()/te.exposure.sum():.1f}')
    preds.append(pd.DataFrame({'row_id':te.row_id,'prediction':pt_metric,'expected_cost':pt_unbiased}))
    out[ln]=dict(oof_d2=D2,calib_a=float(a),calib_c=float(np.exp(c0)),level=lvl,iters=its)
    pd.DataFrame({'row_id':d.row_id,'year':d.year,'y':y,'oof':oof,'exposure':d.exposure}).to_parquet(W+f'oof_v2_{ln}.parquet')
    imp=pd.Series(m.feature_importance('gain'),index=X.columns); (imp/imp.sum()).sort_values(ascending=False).to_csv(W+f'imp_v2_{ln}.csv')
s=pd.concat(preds); s[['row_id','prediction']].to_csv(W+'submission_v2.csv',index=False); s.to_parquet(W+'pred_v2_full.parquet')
json.dump(out,open(W+'cv_results_v2.json','w'),indent=1,default=str); print(s.prediction.describe())
