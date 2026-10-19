import pandas as pd
P='/home/claude/data_raw/public/'
c=pd.read_parquet(P+'claims.parquet')
for k in ['loss_date','report_date']: c[k]=pd.to_datetime(c[k])
c['ly']=c.loss_date.dt.year
c['lag']=(c.report_date-c.loss_date).dt.days
print(c.groupby('line').lag.describe(percentiles=[.5,.9,.99]))
print(c.groupby(['line','status'])[['paid_to_date','incurred_to_date','recoveries']].mean())
print(c.groupby(['line','peril']).agg(n=('claim_id','size'),inc=('incurred_to_date','mean'),mx=('incurred_to_date','max')))
print((c.incurred_to_date<c.paid_to_date).mean(), (c[c.status=='repudiated'].incurred_to_date>0).mean())
print(c[c.status=='closed'].eval('incurred_to_date-paid_to_date').abs().max())
m=pd.read_parquet(P+'policies_motor.parquet'); print(m.dtypes); print(m.head(3).T)
t=pd.read_parquet(P+'test_motor.parquet'); print(t.period_start.min(), t.period_end.max()); print(t.exposure.describe())
print(m.groupby(pd.to_datetime(m.period_start).dt.year).exposure.sum())
s=pd.read_parquet(P+'sample_submission.parquet'); print(s.head(), s.prediction.describe())
