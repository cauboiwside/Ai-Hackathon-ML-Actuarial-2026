"""One-way balance correction (minimum-bias style): for each rating factor, compute held-out A/E per level, apply a
credibility-weighted factor, iterate. Factors chosen per line. Evaluated on OOF (in-sample for the correction, shrunk)."""
import sys, json; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import pandas as pd, numpy as np
from common import d2
W2='/home/claude/iaa-pricing/work_p2/'; E='/home/claude/data_p2/emerging/'
K=float(sys.argv[1]) if len(sys.argv)>1 else 1500.0; ROUNDS=3
FACT={'motor':['region','channel','driver_age_band','cover_type','vehicle_use','overnight_parking','voluntary_excess','payment_method','fuel_type','body_type'],
      'buildings':['region','channel','policyholder_age_band','property_type','construction_type','flood_zone','voluntary_excess','payment_method','occupancy','urban_density'],
      'contents':['region','channel','policyholder_age_band','property_type','tenure','security_level','voluntary_excess','payment_method','urban_density','flood_zone']}
addr=pd.read_parquet(E+'addresses.parquet'); st=json.load(open(W2+'stack_p2_v11b.json'))
v11=pd.read_csv(W2+'p2_submission_v11.csv',dtype={'row_id':str}).set_index('row_id').prediction; out=v11.copy()
for ln,L in [('motor','M'),('buildings','B'),('contents','C')]:
    o=pd.read_parquet(W2+f'oof_{ln}.parquet'); em=pd.read_parquet(W2+f'em_train_{ln}.parquet'); te=pd.read_parquet(W2+f'em_test_{ln}.parquet')
    b=sum(wi*o[c].values*o.y.sum()/o[c].sum() for wi,c in zip(st[ln]['w'],st[ln]['names'])); y=o.y.values; e=o.exposure.values
    pred=b.copy(); ptest=v11[te.row_id].values.copy(); facs={}
    for rd in range(ROUNDS):
        for f in FACT[ln]:
            if f not in em: continue
            lev_tr=em[f].astype(str).values; lev_te=te[f].astype(str).values
            t=pd.DataFrame({'y':y,'p':pred,'e':e,'l':lev_tr}).groupby('l').agg(y=('y','sum'),p=('p','sum'),e=('e','sum'))
            fac=1+t.e/(t.e+K)*(t.y/t.p-1)
            pred*=pd.Series(lev_tr).map(fac).fillna(1).values; ptest*=pd.Series(lev_te).map(fac).fillna(1).values
            facs[f]=fac.round(3).to_dict() if rd==ROUNDS-1 else facs.get(f)
    print(f'{ln}: OOF D2 before {d2(y,b):.4f} after balance {d2(y,pred):.4f} (K={K})')
    for f in ['region','channel','voluntary_excess']: print(f'   {f}: {facs[f]}')
    s=pd.Series(ptest,index=te.row_id.values); nb=te.new_business.values
    for seg in [True,False]:
        ids=te.row_id.values[nb==seg]; out[ids]=s[ids]*v11[ids].sum()/s[ids].sum()
assert (out>=0).all() and np.isfinite(out).all()
name=sys.argv[2] if len(sys.argv)>2 else 'v14'
out.rename('prediction').reset_index().to_csv(W2+f'p2_submission_{name}.csv',index=False); out.rename('prediction').reset_index().to_csv(f'/mnt/user-data/outputs/p2_submission_{name}.csv',index=False)
print('corr with v11',round(out.corr(v11),4))
