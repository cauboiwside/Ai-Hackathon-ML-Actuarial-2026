import pandas as pd, numpy as np
P='/home/claude/data_raw/public/'; W='/home/claude/iaa-pricing/work/'
c=pd.read_parquet(W+'claims_prepped.parquet'); c['ly']=c.loss_date.dt.year; c=c[c.ly>=2021]
bi=c[(c.peril=='bodily_injury')]
print(bi.groupby(['ly','status']).agg(n=('cost','size'),mean=('cost','mean'),med=('cost','median'),paid=('paid_to_date','mean')).round(0))
# report lag by peril, using 2021 losses only (3+ years observed)
c21=c[c.ly==2021]
print(c21.groupby(['line','peril']).lag.describe(percentiles=[.5,.9,.95,.99]).round(0))
w=pd.read_parquet(P+'weather.parquet'); print(w.year.value_counts().sort_index())
for f in ['test_motor','test_buildings','test_contents']:
    t=pd.read_parquet(P+f+'.parquet'); print(f, t.shape, t.columns.tolist()[:8], t.exposure.sum())
