import pandas as pd, numpy as np
E='/home/claude/data_p2/emerging/'; D='/home/claude/data_raw/public/'
pd.set_option('display.width',250)
print('## exposure / dates / duplicates')
for f in ['policies_motor','policies_buildings','policies_contents','test_motor','test_buildings','test_contents']:
    d=pd.read_parquet(E+f+'.parquet'); days=(pd.to_datetime(d.period_end)-pd.to_datetime(d.period_start)).dt.days+1
    r=(d.exposure/(days/365)).round(2).value_counts().head(4).to_dict()
    print(f'{f}: expo/calc {r} | dup row_id {d.row_id.duplicated().sum()} | years {pd.to_datetime(d.period_start).dt.year.value_counts().to_dict()} | new_business {d.new_business.mean():.2f} | channel {d.channel.value_counts(normalize=True).round(2).to_dict()}')
print('\n## categorical levels: emerging vs developed (train + test)')
for f,t in [('policies_motor','test_motor'),('policies_buildings','test_buildings'),('policies_contents','test_contents'),('customers',None),('addresses',None)]:
    d=pd.read_parquet(E+f+'.parquet'); dd=pd.read_parquet(D+f+'.parquet'); te=pd.read_parquet(E+t+'.parquet') if t else None
    for c in d.columns:
        if str(d[c].dtype) in ('object','str') and not c.endswith('_id') and c not in ('period_start','period_end','surname'):
            ev=set(d[c].dropna().astype(str)); dv=set(dd[c].dropna().astype(str)) if c in dd else set(); tv=set(te[c].dropna().astype(str)) if te is not None and c in te else set()
            low=len(ev)!=len({v.strip().lower() for v in ev})
            if ev-dv or tv-ev or low: print(f'  {f}.{c}: emerging-only {sorted(ev-dv)[:10]} | test-only {sorted(tv-ev)[:6]} | case-variants={low} | n={len(ev)}')
print('\n## numeric ranges emerging train vs test vs developed')
for f,t in [('policies_motor','test_motor'),('policies_buildings','test_buildings'),('policies_contents','test_contents')]:
    d=pd.read_parquet(E+f+'.parquet'); te=pd.read_parquet(E+t+'.parquet'); dd=pd.read_parquet(D+f+'.parquet')
    for c in d.columns:
        if d[c].dtype.kind in 'if' and c!='exposure':
            q=lambda s: [round(x,1) for x in s.quantile([0,.5,1]).tolist()]
            a,b=q(d[c]),q(te[c]); cc=q(dd[c]) if c in dd else None
            if a[0]<0 or (b[2]>a[2]*1.5+1) or (cc and (a[2]>cc[2]*3 or a[1]>cc[1]*2.5 or a[1]<cc[1]/2.5)): print(f'  {f}.{c}: train {a} test {b} dev {cc}')
print('\n## claims')
c=pd.read_parquet(E+'claims.parquet'); c['loss_date']=pd.to_datetime(c.loss_date); c['report_date']=pd.to_datetime(c.report_date)
print('loss years',c.loss_date.dt.year.value_counts().to_dict(),'| max report date',c.report_date.max().date(),'| status',c.status.value_counts().to_dict())
print('neg paid',(c.paid_to_date<0).sum(),'inc<paid',(c.incurred_to_date<c.paid_to_date).sum(),'report<loss',(c.report_date<c.loss_date).sum(),'| lag median',(c.report_date-c.loss_date).dt.days.median())
cd=pd.read_parquet(D+'claims.parquet')
print(c.groupby(['line','peril']).incurred_to_date.agg(['size','mean','median']).round(0).join(cd.groupby(['line','peril']).incurred_to_date.agg(['size','mean','median']).round(0),rsuffix='_dev'))
print('perils emerging-only:',set(c.peril)-set(cd.peril),'| dev-only:',set(cd.peril)-set(c.peril))
print('recoveries>0 share',(c.recoveries>0).mean().round(3),'| dev',(cd.recoveries>0).mean().round(3))
