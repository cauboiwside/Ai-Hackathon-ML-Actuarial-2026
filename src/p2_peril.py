"""Per-peril frequency x severity transfer prior (T3) from the developed market on common features, + offset local model O3.
Scored on the same emerging folds; appended to oof/pred; re-stacks all components -> stack_p2_v5.json."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PL=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2)
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2,seed=9)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2,seed=9)
ITF={'motor':{'bodily_injury':220,'glass_hail':290,'own_damage':330,'theft':230,'third_party_prop':350},'buildings':{'accidental_damage':160,'escape_of_water':190,'fire':35,'flood':190,'storm':220,'subsidence':140},'contents':{'accidental_damage':200,'escape_of_water':95,'flood':170,'theft':165}}
ITS={'motor':{'bodily_injury':15,'glass_hail':45,'own_damage':130,'theft':55,'third_party_prop':60},'buildings':{'accidental_damage':25,'escape_of_water':60,'fire':90,'flood':60,'storm':60,'subsidence':75},'contents':{'accidental_damage':50,'escape_of_water':55,'flood':100,'theft':70}}
def mk(df,cats=None,drop=()):
    X=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    if cats is None: cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':np.nan,'None':np.nan}).astype('category')
    return X,cats
def align(df,X,cats,drop=()):
    Xt=df.drop(columns=[c for c in ID+EXCL if c in df]+list(drop))
    for c in cats: Xt[c]=pd.Categorical(Xt[c].astype(str).replace({'nan':np.nan,'None':np.nan}),categories=X[c].cat.categories)
    return Xt[X.columns]
cl=pd.read_parquet('/home/claude/iaa-pricing/work/claims_level.parquet')[['claim_id','cost_c','peril']]
for ln in (sys.argv[1].split(',') if len(sys.argv)>1 else ['motor','buildings','contents']):
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    j=pd.read_parquet(W2+f'claimrows_dev_{ln}.parquet')  # claim_id,row_id,cost_c,peril
    X,cats=mk(dv,drop=('region',)); Xe=align(em,X,cats,('region',)); Xte=align(te,X,cats,('region',)); off=np.log(dv.exposure.values)
    T3_em=np.zeros(len(em)); T3_te=np.zeros(len(te))
    for peril,jp in j[j.cost_c>0].groupby('peril'):
        n=dv.row_id.map(jp.groupby('row_id').size()).fillna(0).astype(int).values; cst=dv.row_id.map(jp.groupby('row_id').cost_c.sum()).fillna(0).values; has=n>0
        br=n.sum()/dv.exposure.sum(); offp=off+np.log(br)
        mf=lgb.train(PF,lgb.Dataset(X,n,init_score=offp,categorical_feature=cats),ITF[ln][peril])
        ms=lgb.train(PS,lgb.Dataset(X[has],cst[has]/n[has],weight=n[has],categorical_feature=cats),ITS[ln][peril])
        T3_em+=em.exposure.values*br*np.exp(mf.predict(Xe,raw_score=True))*ms.predict(Xe); T3_te+=te.exposure.values*br*np.exp(mf.predict(Xte,raw_score=True))*ms.predict(Xte)
        print(f'  {ln}/{peril} done',flush=True)
    ye=em.target.values; l3=ye.sum()/T3_em.sum(); T3_em*=l3; T3_te*=l3
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    print(f'{ln}: T3 per-peril transfer D2={d2(ye,T3_em):.4f} | T2={d2(ye,o["T2"].values):.4f}',flush=True)
    XL,catsL=mk(em); XLt=align(te,XL,catsL); folds=list(GroupKFold(5).split(em,groups=em.policy_id))
    oof=np.zeros(len(em)); pt=np.zeros(len(te)); seeds=[1,2,3,4,5]
    for sd in seeds:
        its=[]
        for a,b in folds:
            m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL.iloc[a],ye[a],init_score=np.log(T3_em)[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=np.log(T3_em)[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[b]+=np.exp(m.predict(XL.iloc[b],raw_score=True)+np.log(T3_em)[b])/len(seeds); its.append(m.best_iteration)
        m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL,ye,init_score=np.log(T3_em),categorical_feature=catsL),max(10,int(np.mean(its)*1.1)))
        pt+=np.exp(m.predict(XLt,raw_score=True)+np.log(T3_te))/len(seeds)
    print(f'   O3 (offset on T3, 5-seed)={d2(ye,oof):.4f} | O2={d2(ye,o["O2"].values):.4f}',flush=True)
    o['T3']=T3_em; o['O3']=oof; p['T3']=T3_te; p['O3']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
    names=['T','L','O','P','O5','O2','T2','P3','T3','O3']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(w): w=np.abs(w)/np.abs(w).sum(); return sum(wi*c for wi,c in zip(w,comps))
    r=minimize(lambda w: tweedie_dev(ye,blend(w)).sum(),[0.1]*10,method='Nelder-Mead',options={'maxiter':6000}); w=np.abs(r.x)/np.abs(r.x).sum()
    v4=json.load(open(W2+'stack_p2_v4.json'))[ln]['d2']
    print(f'   STACK v5 {dict(zip(names,np.round(w,2)))} OOF D2={d2(ye,blend(w)):.4f} (v4 {v4:.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v5.json')) if os.path.exists(W2+'stack_p2_v5.json') else {}
    st[ln]=dict(names=names,w=w.tolist(),d2=float(d2(ye,blend(w)))); json.dump(st,open(W2+'stack_p2_v5.json','w'),indent=1)
print('done')
