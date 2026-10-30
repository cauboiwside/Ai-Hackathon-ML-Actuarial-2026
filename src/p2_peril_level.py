"""T4: per-peril transfer prior with per-peril LEVEL transfer: each peril's developed frequency and severity are
re-levelled to the emerging market's own per-peril experience (credibility-weighted, Buhlmann-style K), instead of one
level factor per line. O4 = 5-seed local offset model on T4. Appends T4/O4, re-stacks -> stack_p2_v9.json."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
PL=dict(objective='tweedie',tweedie_variance_power=1.5,learning_rate=0.03,num_leaves=8,min_data_in_leaf=300,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=30,cat_smooth=100,min_data_per_group=300,cat_l2=30,verbose=-1,num_threads=2)
PF=dict(objective='poisson',learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=200,cat_l2=10,verbose=-1,num_threads=2)
PS=dict(objective='gamma',learning_rate=0.03,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,cat_smooth=50,min_data_per_group=100,cat_l2=10,verbose=-1,num_threads=2)
ITF={'motor':{'bodily_injury':220,'glass_hail':290,'own_damage':330,'theft':230,'third_party_prop':350},'buildings':{'accidental_damage':160,'escape_of_water':190,'fire':35,'flood':190,'storm':220,'subsidence':140},'contents':{'accidental_damage':200,'escape_of_water':95,'flood':170,'theft':165}}
ITS={'motor':{'bodily_injury':15,'glass_hail':45,'own_damage':130,'theft':55,'third_party_prop':60},'buildings':{'accidental_damage':25,'escape_of_water':60,'fire':90,'flood':60,'storm':60,'subsidence':75},'contents':{'accidental_damage':50,'escape_of_water':55,'flood':100,'theft':70}}
K_FREQ=2000; K_SEV=300     # credibility constants (exposure-years for frequency, claims for severity)
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
    jd=pd.read_parquet(W2+f'claimrows_dev_{ln}.parquet'); je=pd.read_parquet(W2+f'claimrows_em_{ln}.parquet')
    jd=jd[jd.cost_c>0]; je=je[je.cost_c>0]
    X,cats=mk(dv,drop=('region',)); Xe=align(em,X,cats,('region',)); Xte=align(te,X,cats,('region',)); off=np.log(dv.exposure.values)
    T4_em=np.zeros(len(em)); T4_te=np.zeros(len(te)); ye=em.target.values
    for peril,jp in jd.groupby('peril'):
        n=dv.row_id.map(jp.groupby('row_id').size()).fillna(0).astype(int).values; cst=dv.row_id.map(jp.groupby('row_id').cost_c.sum()).fillna(0).values; has=n>0
        br=n.sum()/dv.exposure.sum(); offp=off+np.log(br); sev_dev=cst.sum()/n.sum()
        mf=lgb.train({**PF,'seed':7},lgb.Dataset(X,n,init_score=offp,categorical_feature=cats),ITF[ln][peril])
        ms=lgb.train({**PS,'seed':7},lgb.Dataset(X[has],cst[has]/n[has],weight=n[has],categorical_feature=cats),ITS[ln][peril])
        f_em=br*np.exp(mf.predict(Xe,raw_score=True)); s_em=ms.predict(Xe)         # developed-relativity frequency rate and severity on emerging rows
        # per-peril level transfer with credibility: local observed vs developed-model predicted, on the emerging training rows
        jpe=je[je.peril==peril]; n_loc=len(jpe); cost_loc=jpe.cost_c.sum(); E=em.exposure.sum()
        freq_ratio=(n_loc+K_FREQ*(f_em*em.exposure.values).sum()/E)/( (f_em*em.exposure.values).sum()+K_FREQ*(f_em*em.exposure.values).sum()/E )   # = credibility blend of local/model frequency
        Z_f=E/(E+K_FREQ); freq_ratio=Z_f*(n_loc/(f_em*em.exposure.values).sum())+(1-Z_f)*1.0
        # severity: local mean severity vs model mean severity over the rows that had that peril
        rows=em.row_id.isin(jpe.row_id).values; model_sev=s_em[rows].mean() if rows.any() else sev_dev
        Z_s=n_loc/(n_loc+K_SEV); sev_ratio=Z_s*((cost_loc/max(n_loc,1))/model_sev)+(1-Z_s)*1.0 if n_loc>0 else 1.0
        T4_em+=em.exposure.values*f_em*freq_ratio*s_em*sev_ratio
        T4_te+=te.exposure.values*br*np.exp(mf.predict(Xte,raw_score=True))*freq_ratio*ms.predict(Xte)*sev_ratio
        print(f'  {ln}/{peril}: local claims {n_loc}, freq ratio {freq_ratio:.2f} (Z={Z_f:.2f}), sev ratio {sev_ratio:.2f} (Z={Z_s:.2f})',flush=True)
    l4=ye.sum()/T4_em.sum(); T4_em*=l4; T4_te*=l4
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    print(f'{ln}: T4 per-peril-level transfer D2={d2(ye,T4_em):.4f} | T3={d2(ye,o["T3"].values):.4f} T2={d2(ye,o["T2"].values):.4f} | residual line level {l4:.3f}',flush=True)
    XL,catsL=mk(em); XLt=align(te,XL,catsL); folds=list(GroupKFold(5).split(em,groups=em.policy_id))
    oof=np.zeros(len(em)); pt=np.zeros(len(te)); seeds=[1,2,3,4,5]; lo=np.log(T4_em); lt=np.log(T4_te)
    for sd in seeds:
        its=[]
        for a,b in folds:
            m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL.iloc[a],ye[a],init_score=lo[a],categorical_feature=catsL),2000,valid_sets=[lgb.Dataset(XL.iloc[b],ye[b],init_score=lo[b],categorical_feature=catsL)],callbacks=[lgb.early_stopping(100,verbose=False)])
            oof[b]+=np.exp(m.predict(XL.iloc[b],raw_score=True)+lo[b])/len(seeds); its.append(m.best_iteration)
        m=lgb.train({**PL,'seed':sd},lgb.Dataset(XL,ye,init_score=lo,categorical_feature=catsL),max(10,int(np.mean(its)*1.1)))
        pt+=np.exp(m.predict(XLt,raw_score=True)+lt)/len(seeds)
    print(f'   O4 (offset on T4, 5-seed)={d2(ye,oof):.4f} | O3={d2(ye,o["O3"].values):.4f} O2={d2(ye,o["O2"].values):.4f}',flush=True)
    o['T4']=T4_em; o['O4']=oof; p['T4']=T4_te; p['O4']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
    names=['T','L','O','P','O5','O2','T2','P3','T3','O3','PF2','T4','O4']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(wv): wv=np.abs(wv)/np.abs(wv).sum(); return sum(wi*c for wi,c in zip(wv,comps))
    r=minimize(lambda wv: tweedie_dev(ye,blend(wv)).sum(),[0.1]*len(names),method='Nelder-Mead',options={'maxiter':10000}); wv=np.abs(r.x)/np.abs(r.x).sum()
    v6=json.load(open(W2+'stack_p2_v6.json'))[ln]['d2']
    print(f'   STACK v9 {dict(zip(names,np.round(wv,2)))} OOF D2={d2(ye,blend(wv)):.4f} (v6 {v6:.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v9.json')) if os.path.exists(W2+'stack_p2_v9.json') else {}
    st[ln]=dict(names=names,w=wv.tolist(),d2=float(d2(ye,blend(wv)))); json.dump(st,open(W2+'stack_p2_v9.json','w'),indent=1)
print('done')
