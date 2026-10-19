import pandas as pd, numpy as np
W='/home/claude/iaa-pricing/work/'
c=pd.read_parquet(W+'claims_prepped.parquet'); c['ly']=c.loss_date.dt.year
c=c[c.ly>=2021]
print(c.nlargest(8,'cost')[['line','peril','ly','status','cost','paid_to_date']])
nr=c[c.status!='repudiated']
t=nr.pivot_table(index=['line','peril'],columns='ly',values='cost',aggfunc='median').round(0); print('median sev\n',t)
t=nr[nr.cost<200000].pivot_table(index=['line','peril'],columns='ly',values='cost',aggfunc='mean').round(0); print('mean sev (<200k)\n',t)
t=nr.pivot_table(index=['line','peril'],columns='ly',values='claim_id',aggfunc='size'); print(t)
# open share and open vs closed severity by peril for 2023/2024
print(nr.groupby(['line','peril','ly']).apply(lambda g: pd.Series({'open%':(g.status=='open').mean(),'open_mean':g[g.status=='open'].cost.mean(),'closed_mean':g[g.status=='closed'].cost.mean()}),include_groups=False).unstack('ly').round(2))
# settlement: for closed, we don't have close date. report lag vs size
print(nr.groupby(['line',pd.cut(nr.lag,[0,7,30,90,365,5000])],observed=True).cost.mean().unstack().round(0))
