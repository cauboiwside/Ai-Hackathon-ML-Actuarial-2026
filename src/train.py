"""Step 4: grouped CV (by policy), A/E by year -> time calibration, final fit, test predictions."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import *
PRM=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=31,min_data_in_leaf=500,
         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,
         cat_l2=10,verbose=-1,num_threads=2,seed=1)
lp=json.load(open(W+'level_params.json')); out={}; preds=[]
for ln in ['motor','buildings','contents']:
    d=pd.read_parquet(W+f'mf_train_{ln}.parquet'); te=pd.read_parquet(W+f'mf_test_{ln}.parquet')
    X,cats=prep(d); Xt,_=prep(te,cats)
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str),categories=X[c].cat.categories)
    Xt=Xt[X.columns]
    y=d.target.values; off=np.log(d.exposure.values); oof=np.zeros(len(d)); its=[]
    for k,(a,b) in enumerate(GroupKFold(4).split(X,y,d.policy_id)):
        ds=lgb.Dataset(X.iloc[a],y[a],init_score=off[a],categorical_feature=cats)
        dv=lgb.Dataset(X.iloc[b],y[b],init_score=off[b],categorical_feature=cats,reference=ds)
        m=lgb.train(PRM,ds,3000,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
        oof[b]=np.exp(m.predict(X.iloc[b],raw_score=True)+off[b]); its.append(m.best_iteration)
    ae=pd.DataFrame({'y':y,'p':oof,'yr':d.year}).groupby('yr').sum(); ae['AE']=ae.y/ae.p
    # time trend in A/E, weighted log-linear, projected to 2025
    b1,b0=np.polyfit(ae.index,np.log(ae.AE),1,w=np.sqrt(ae.p))
    cal=float(np.exp(b0+b1*2025))
    D2=d2(y,oof); D2y={int(yr):round(d2(y[d.year==yr],oof[d.year==yr]),4) for yr in ae.index}
    print(f'{ln}: iters={its} OOF D2={D2:.4f} by year={D2y}'); print(ae.AE.round(3).to_dict(),'-> 2025 A/E',round(cal,3),'annual drift',round(np.exp(b1)-1,4))
    ds=lgb.Dataset(X,y,init_score=off,categorical_feature=cats)
    m=lgb.train(PRM,ds,int(np.mean(its)*1.1))
    m.save_model(W+f'lgb_{ln}.txt')
    pt=np.exp(m.predict(Xt,raw_score=True))*te.exposure.values
    pt=pt*cal*lp['ll'][ln]
    preds.append(pd.DataFrame({'row_id':te.row_id,'prediction':pt}))
    out[ln]=dict(oof_d2=D2,d2_by_year=D2y,ae=ae.AE.to_dict(),cal_2025=cal,ll=lp['ll'][ln],iters=its,
                 test_pp=float(pt.sum()/te.exposure.sum()),train_pp=float(y.sum()/d.exposure.sum()))
    print('  test pp',round(out[ln]['test_pp'],2),'train pp (2025 level)',round(out[ln]['train_pp'],2))
    imp=pd.Series(m.feature_importance('gain'),index=X.columns); (imp/imp.sum()).sort_values(ascending=False).to_csv(W+f'imp_{ln}.csv')
    pd.DataFrame({'row_id':d.row_id,'year':d.year,'y':y,'oof':oof,'exposure':d.exposure}).to_parquet(W+f'oof_{ln}.parquet')
sub=pd.concat(preds); json.dump(out,open(W+'cv_results.json','w'),indent=1,default=str)
sub.to_csv(W+'submission_v1.csv',index=False); print(sub.shape, sub.prediction.describe())
