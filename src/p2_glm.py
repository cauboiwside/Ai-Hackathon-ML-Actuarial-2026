"""GLM benchmark / blend component: Tweedie(1.5) log-link GLM on emerging rows, one-hot categoricals, numeric features
standardised, log(T2 prior rate) as a feature (credibility-like), elastic regularisation. Same GroupKFold(5)."""
import sys, json, os; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import numpy as np, pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
from common import tweedie_dev, d2
W2='/home/claude/iaa-pricing/work_p2/'
ID=['row_id','policy_id','customer_id','address_id','target','n_claims','year','exposure']
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment','email_domain','marketing_consent','document_preference','branch_id']
def design(df,prior_rate,cols=None,stats=None):
    X=df.drop(columns=[c for c in ID+EXCL+['make_model','occupation'] if c in df]).copy()   # high-cardinality: use developed TE priors instead
    cats=[c for c in X.columns if str(X[c].dtype) in ('object','str','category','bool')]
    for c in cats: X[c]=X[c].astype(str).replace({'nan':'NA','None':'NA'})
    num=[c for c in X.columns if c not in cats]
    X[num]=X[num].fillna(X[num].median() if stats is None else stats['med'])
    X=pd.get_dummies(X,columns=cats,dtype=float)
    X['log_prior']=np.log(prior_rate)
    if cols is None:
        cols=X.columns; mu=X[num+['log_prior']].mean(); sd=X[num+['log_prior']].std().replace(0,1); stats={'med':df[num].median(),'mu':mu,'sd':sd}
    X=X.reindex(columns=cols,fill_value=0.0)
    X[num+['log_prior']]=(X[num+['log_prior']]-stats['mu'])/stats['sd']
    return X.values.astype(float),cols,stats
for ln in ['motor','buildings','contents']:
    em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet'); o=pd.read_parquet(W2+f'oof_{ln}.parquet'); p=pd.read_parquet(W2+f'pred_{ln}.parquet')
    dv=pd.read_parquet(W2+f'dev_train_{ln}.parquet')
    for c in ['make_model','occupation']:
        if c in dv:
            mean=dv.target.sum()/dv.exposure.sum(); g_=dv.groupby(c,observed=True).agg(y=('target','sum'),e=('exposure','sum')); mp=((g_.y+200*mean)/(g_.e+200)/mean).apply(np.log)
            for df in (em,te): df['devte_'+c]=df[c].astype(str).str.strip().map(mp).fillna(0.0)
    ye=em.target.values; e=em.exposure.values; rate=ye/e
    oof=np.zeros(len(em))
    for a,b in GroupKFold(5).split(em,groups=em.policy_id):
        Xa,cols,st=design(em.iloc[a],o['T2'].values[a]/e[a]); Xb,_,_=design(em.iloc[b],o['T2'].values[b]/e[b],cols,st)
        g=TweedieRegressor(power=1.5,link='log',alpha=0.05,max_iter=300).fit(Xa,rate[a],sample_weight=e[a])
        oof[b]=g.predict(Xb)*e[b]
    Xa,cols,st=design(em,o['T2'].values/e); Xt,_,_=design(te,p['T2'].values/te.exposure.values,cols,st)
    g=TweedieRegressor(power=1.5,link='log',alpha=0.05,max_iter=300).fit(Xa,rate,sample_weight=e); pt=g.predict(Xt)*te.exposure.values
    print(f'{ln}: GLM OOF D2={d2(ye,oof*ye.sum()/oof.sum()):.4f} (O2 {d2(ye,o["O2"].values):.4f}) | {Xa.shape[1]} coefficients',flush=True)
    o['GLM']=oof; p['GLM']=pt; o.to_parquet(W2+f'oof_{ln}.parquet'); p.to_parquet(W2+f'pred_{ln}.parquet')
    st6=json.load(open(W2+'stack_p2_v6.json'))[ln]; names=st6['names']+['GLM']; comps=[o[c].values*ye.sum()/o[c].sum() for c in names]
    def blend(wv): wv=np.abs(wv)/np.abs(wv).sum(); return sum(wi*c for wi,c in zip(wv,comps))
    w0=np.array(st6['w']+[0.1]); r=minimize(lambda wv: tweedie_dev(ye,blend(wv)).sum(),w0,method='Nelder-Mead',options={'maxiter':8000}); wv=np.abs(r.x)/np.abs(r.x).sum()
    print(f'   STACK +GLM: GLM weight {wv[-1]:.2f}, OOF D2={d2(ye,blend(wv)):.4f} (v6 stack {st6["d2"]:.4f})',flush=True)
    st=json.load(open(W2+'stack_p2_v11.json')) if os.path.exists(W2+'stack_p2_v11.json') else {}
    st[ln]=dict(names=names,w=wv.tolist(),d2=float(d2(ye,blend(wv)))); json.dump(st,open(W2+'stack_p2_v11.json','w'),indent=1)
    coef=pd.Series(g.coef_,index=cols).sort_values(); print('   largest GLM effects:',', '.join(f'{k}:{v:+.2f}' for k,v in pd.concat([coef.head(4),coef.tail(4)]).items()))
print('done')
