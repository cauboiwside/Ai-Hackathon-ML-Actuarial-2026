import pandas as pd, numpy as np
W='/home/claude/iaa-pricing/work/'
for ln in ['motor','buildings']:
    d=pd.read_parquet(W+f'mf_train_{ln}.parquet'); t=pd.read_parquet(W+f'mf_test_{ln}.parquet')
    d['has']=d.target>0
    print(ln,'contacts: train vs test dist'); print(pd.concat([d.customer_service_contacts.value_counts(normalize=True).head(8),t.customer_service_contacts.value_counts(normalize=True).head(8)],axis=1).round(4))
    print(d.groupby(d.customer_service_contacts.clip(upper=6)).agg(n=('has','size'),claim_rate=('has','mean'),pp=('target','mean')).round(3))
    print(d.groupby('previous_insurer').agg(n=('has','size'),claim_rate=('has','mean'),pp=('target','mean')).sort_values('n').round(3).tail(12))
    print('prev insurer test share', t.previous_insurer.value_counts(normalize=True).round(3).head(12).to_dict())
    print(d.groupby('occupation').agg(n=('has','size'),cr=('has','mean')).sort_values('cr').round(3).iloc[[0,1,2,-3,-2,-1]])
