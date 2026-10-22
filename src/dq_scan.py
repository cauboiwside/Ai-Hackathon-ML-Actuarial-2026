import pandas as pd, numpy as np
P='/home/claude/data_raw/public/'
for f,t in [('policies_motor','test_motor'),('policies_buildings','test_buildings'),('policies_contents','test_contents')]:
    d=pd.read_parquet(P+f+'.parquet'); te=pd.read_parquet(P+t+'.parquet')
    d['days']=(pd.to_datetime(d.period_end)-pd.to_datetime(d.period_start)).dt.days+1
    d['exp_calc']=d.days/365
    bad=(d.exposure/d.exp_calc).round(2)
    print(f'\n== {f}: exposure/calc ratio counts:',bad.value_counts().head(6).to_dict())
    print('   dup row_id',d.row_id.duplicated().sum(),'| exact dup rows (excl row_id)',d.drop(columns='row_id').duplicated().sum(),'| period_end<start',(d.days<=0).sum())
    num=[c for c in d.columns if d[c].dtype.kind in 'if' and c not in ('days','exp_calc')]
    for c in num:
        q=d[c].quantile([0,.001,.5,.999,1]).round(2).tolist(); tq=te[c].quantile([0,.001,.5,.999,1]).round(2).tolist() if c in te else None
        flag=''
        if tq and (q[0]<tq[0]*0.5-1 or q[4]>tq[4]*2+1): flag='  <-- train range beyond test'
        if (d[c]<0).any(): flag+=' NEG'
        if d[c].isna().mean()>0 : flag+=f' na={d[c].isna().mean():.3f}'
        if flag: print(f'   {c:28s} train[min,.1%,med,99.9%,max]={q} test={tq}{flag}')
    for c in [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=='str']:
        if c in ('row_id','policy_id','customer_id','address_id','period_start','period_end'): continue
        tr=set(d[c].dropna().unique()); ts=set(te[c].dropna().unique()) if c in te else set()
        if ts and (tr-ts or ts-tr): print(f'   {c}: train-only levels {sorted(tr-ts)[:8]} test-only {sorted(ts-tr)[:8]}')
c=pd.read_parquet(P+'customers.parquet'); a=pd.read_parquet(P+'addresses.parquet')
print('\ncustomers dob range',c.date_of_birth.min(),c.date_of_birth.max(),'| dup customer_id',c.customer_id.duplicated().sum())
print('addresses year_built',a.year_built.describe()[['min','max']].to_dict(),'| dup',a.address_id.duplicated().sum())
cl=pd.read_parquet(P+'claims.parquet'); print('claims: neg paid',(cl.paid_to_date<0).sum(),'inc<paid',(cl.incurred_to_date<cl.paid_to_date).sum(),'report<loss',(pd.to_datetime(cl.report_date)<pd.to_datetime(cl.loss_date)).sum(),'dup claim',cl.claim_id.duplicated().sum(), 'loss<2021',(pd.to_datetime(cl.loss_date)<'2021-01-01').sum())
