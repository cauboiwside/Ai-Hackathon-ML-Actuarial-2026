"""OP: credibility per peril. For each peril: developed per-peril frequency model (common features) gives the prior rate;
a local Poisson GBM is trained on the emerging rows with offset = log(prior count) (learns local deviations in frequency
relativities); severity = developed per-peril severity model x credibility-weighted local severity level.
Expected cost = sum over perils. GroupKFold(5), 3 seeds for the local frequency models."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=11)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=11)
PLF=dict(objective='poisson',learning_rate=0.03,num_leaves=6,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2)
ITF={'motor':{'bodily_injury':220,'glass_hail':290,'own_damage':330,'theft':230,'third_party_prop':350},'buildings':{'accidental_damage':160,'escape_of_water':190,'fire':35,'flood':190,'storm':220,'subsidence':140},'contents':{'accidental_damage':200,'escape_of_water':95,'flood':170,'theft':165}}
ITS={'motor':{'bodily_injury':15,'glass_hail':45,'own_damage':130,'theft':55,'third_party_prop':60},'buildings':{'accidental_damage':25,'escape_of_water':60,'fire':90,'flood':60,'storm':60,'subsidence':75},'contents':{'accidental_damage':50,'escape_of_water':55,'flood':100,'theft':70}}
K_SEV=300
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
    jd=pd.read_parquet(W2+f'claimrows_dev_{ln}.parquet'); je=pd.read_parquet(W2+f'claimrows_em_{ln}.parquet'); jd=jd[jd.cost_c>0]; je=je[je.cost_c>0]
    X,cats=mk(dv,drop=('region',)); Xe=align(em,X,cats,('region',)); Xte=align(te,X,cats,('region',)); off=np.log(dv.exposure.values)
    XL,catsL=mk(em); XLt=align(te,XL,catsL); folds=list(GroupKFold(5).split(em,groups=em.policy_id)); ye=em.target.values
    oof=np.zeros(len(em)); pt=np.zeros(len(te))
    for peril,jp in jd.groupby('peril'):
        n=dv.row_id.map(jp.groupby('row_id').size()).fillna(0).astype(int).values; cst=dv.row_id.map(jp.groupby('row_id').cost_c.sum()).fillna(0).values; has=n>0
        br=n.sum()/dv.exposure.sum(); offp=off+np.log(br)
        mf=lgb.train(PF,lgb.Dataset(X,n,init_score=offp,categorical_feature=cats),ITF[ln][peril])
        ms=lgb.train(PS,lgb.Dataset(X[has],cst[has]/n[has],weight=n[has],categorical_feature=cats),ITS[ln][peril])
        prior_em=em.exposure.values*br*np.exp(mf.predict(Xe,raw_score=True)); prior_te=te.exposure.values*br*np.exp(mf.predict(Xte,raw_score=True))
        sev_em=ms.predict(Xe); sev_te=ms.predict(Xte)
        # local frequency: Poisson on local per-peril counts with the prior as offset
        jpe=je[je.peril==peril]; n_loc=em.row_id.map(jpe.groupby('row_id').size()).fillna(0).astype(int).values
        lvl=n_loc.sum()/prior_em.sum(); lo=np.log(prior_em*lvl); lt=np.log(prior_te*lvl)      # re-level prior to local peril frequency first
        f_oof=np.zeros(len(em)); f_te=np.zeros(len(te)); seeds=[1,2,3]
        for sd in seeds:
            its=[]
            for a,b in folds:
                m=lgb.train({**PLF,'seed':sd},lgb.Dataset(XL.iloc[a],n_loc[a],init_score=lo[a],categorical_feature=catsL),1500,valid_sets=[lgb.Dataset(XL.iloc[b],n_loc[b],init_score=lo[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(80,verbose=False)])
                f_oof[b]+=np.exp(m.predict(XL.iloc[b],raw_score=True)+lo[b])/len(seeds); its.append(m.best_iteration)
            m=lgb.train({**PLF,'seed':sd},lgb.Dataset(XL,n_loc,init_score=lo,categorical_feature=catsL),max(5,int(np.mean(its)*1.1)))
            f_te+=np.exp(m.predict(XLt,raw_score=True)+lt)/len(seeds)
        # severity level: credibility blend of local mean severity vs developed-model mean severity on claim rows
        rows=n_loc>0; model_sev=(sev_em[rows]*n_loc[rows]).sum()/n_loc[rows].sum() if rows.any() else 1.0
        Z=n_loc.sum()/(n_loc.sum()+K_SEV); sev_ratio=Z*(jpe.cost_c.sum()/max(n_loc.sum(),1)/model_sev)+(1-Z)
        oof+=f_oof*sev_em*sev_ratio; pt+=f_te*sev_te*sev_ratio
        print(f'  {ln}/{peril}: local claims {int(n_loc.sum())}, local freq iters ~{int(np.mean(its))}, sev ratio {sev_ratio:.2f}',flush=True)
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    print(f'{ln}: OP credibility-per-peril OOF D2={d2(ye,oof*ye.sum()/oof.sum()):.4f} | best previous single: O2 {d2(ye,o["O2"].values):.4f} O4 {d2(ye,o["O4"].values):.4f} PF2 {d2(ye,o["PF2"].values):.4f}',flush=True)
    o['OP']=oof; p['OP']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
    base=json.load(open(W2+'stack_p2_v11b.json'))[ln]; names=base['names']+['OP']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(wv): wv=np.abs(wv)/np.abs(wv).sum(); return sum(wi*c for wi,c in zip(wv,comps))
    r=minimize(lambda wv: tweedie_dev(ye,blend(wv)).sum(),np.array(base['w']+[0.15]),method='Nelder-Mead',options={'maxiter':8000}); wv=np.abs(r.x)/np.abs(r.x).sum()
    print(f'   STACK +OP: OP weight {wv[-1]:.2f}, OOF D2={d2(ye,blend(wv)):.4f} (v11 stack {d2(ye,sum(wi*c for wi,c in zip(base["w"],comps[:-1]))):.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v12.json')) if os.path.exists(W2+'stack_p2_v12.json') else {}
    st[ln]=dict(names=names,w=wv.tolist(),d2=float(d2(ye,blend(wv)))); json.dump(st,open(W2+'stack_p2_v12.json','w'),indent=1)
print('done')
