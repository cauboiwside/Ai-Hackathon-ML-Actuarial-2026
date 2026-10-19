import numpy as np, pandas as pd
W='/home/claude/iaa-pricing/work/'
ID=['row_id','policy_id','customer_id','address_id','target','renewal_decision','year','exposure']
# Excluded on governance grounds (see docs/feature_decisions.md):
#  customer_service_contacts - recorded during the period, driven by claims (target leakage)
#  written_premium - current price; circular to price off it
#  marketing/admin fields - no causal link to loss, high-cardinality noise
EXCL=['customer_service_contacts','written_premium','previous_insurer','campaign_code','marketing_segment',
      'email_domain','marketing_consent','document_preference','branch_id']
DROP=ID+EXCL
def tweedie_dev(y,mu,p=1.5):
    mu=np.maximum(mu,1e-9)
    return 2*(np.power(np.maximum(y,0),2-p)/((1-p)*(2-p)) - y*np.power(mu,1-p)/(1-p) + np.power(mu,2-p)/(2-p))
def d2(y,mu,p=1.5):
    return 1-tweedie_dev(y,mu,p).sum()/tweedie_dev(y,np.full_like(y,y.mean(),dtype=float),p).sum()
def prep(df,cats=None):
    X=df.drop(columns=[c for c in DROP if c in df.columns])
    if cats is None: cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype) in ('str','string','category','bool')]
    for c in cats: X[c]=X[c].astype(str).astype('category')
    return X,cats
