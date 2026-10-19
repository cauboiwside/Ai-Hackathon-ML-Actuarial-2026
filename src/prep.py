"""Step 1: attach claims to policy rows, compute claim target, IBNR weights."""
import pandas as pd, numpy as np
P='/home/claude/data_raw/public/'; OUT='/home/claude/iaa-pricing/work/'
import os; os.makedirs(OUT,exist_ok=True)
VAL=pd.Timestamp('2024-12-31')
c=pd.read_parquet(P+'claims.parquet')
for k in ['loss_date','report_date']: c[k]=pd.to_datetime(c[k])
c['lag']=(c.report_date-c.loss_date).dt.days
# Target per claim: incurred net of recoveries, zero if declined
c['cost']=np.where(c.status=='repudiated',0.0,(c.incurred_to_date-c.recoveries).clip(lower=0))
# Report-lag CDF per line from mature loss years (2021-2022 losses: >=2y observed)
lagcdf={}
for ln,g in c[c.loss_date.dt.year.isin([2021,2022])].groupby('line'):
    lags=np.sort(g.lag.values); lagcdf[ln]=lags
def F(ln,days):
    l=lagcdf[ln]; return np.searchsorted(l,days,side='right')/len(l)
c['age']=(VAL-c.loss_date).dt.days
c['F']=np.nan
for ln in lagcdf: m=c.line==ln; c.loc[m,'F']=F(ln,c.loc[m,'age'].values)
c['F']=c.F.clip(lower=0.25)
c['w_ibnr']=1/c.F
c.to_parquet(OUT+'claims_prepped.parquet')
for ln in ['motor','buildings','contents']:
    print(ln, 'F at 0,30,90,180,365 days:', [round(F(ln,d),3) for d in [0,30,90,180,365]])
# Attach to policy rows
for ln,f in [('motor','policies_motor'),('buildings','policies_buildings'),('contents','policies_contents')]:
    p=pd.read_parquet(P+f+'.parquet')
    p['ps']=pd.to_datetime(p.period_start); p['pe']=pd.to_datetime(p.period_end)
    cl=c[c.line==ln][['claim_id','policy_id','loss_date','cost','w_ibnr','status','peril']]
    j=cl.merge(p[['row_id','policy_id','ps','pe']],on='policy_id',how='left')
    j=j[(j.loss_date>=j.ps)&(j.loss_date<=j.pe)]
    dup=j.claim_id.duplicated().sum()
    j=j.drop_duplicates('claim_id')
    print(ln,'claims',len(cl),'matched',len(j),'unmatched',len(cl)-len(j),'dup-matches',dup)
    j['cost_ult']=j.cost*j.w_ibnr
    agg=j.groupby('row_id').agg(n_claims=('claim_id','size'),cost=('cost','sum'),cost_ult=('cost_ult','sum'),
                                n_ult=('w_ibnr','sum'))
    p=p.merge(agg,on='row_id',how='left').fillna({'n_claims':0,'cost':0,'cost_ult':0,'n_ult':0})
    p['year']=p.ps.dt.year
    s=p.groupby('year').agg(expo=('exposure','sum'),n=('n_claims','sum'),n_ult=('n_ult','sum'),cost=('cost','sum'),cost_ult=('cost_ult','sum'))
    s['freq']=s.n_ult/s.expo; s['sev']=s.cost_ult/s.n_ult; s['pp']=s.cost_ult/s.expo; s['pp_rep']=s.cost/s.expo
    print(s.round(4))
    j.to_parquet(OUT+f'claimrows_{ln}.parquet')
    p.drop(columns=['ps','pe']).to_parquet(OUT+f'train_{ln}.parquet')
