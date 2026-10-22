"""Step 3: build model frames (train with adjusted target, test) per line."""
import pandas as pd, numpy as np
P='/home/claude/data_raw/public/'; W='/home/claude/iaa-pricing/work/'
cust=pd.read_parquet(P+'customers.parquet').drop(columns=['surname'])
cust['date_of_birth']=pd.to_datetime(cust.date_of_birth)
addr=pd.read_parquet(P+'addresses.parquet')
wx=pd.read_parquet(P+'weather.parquet')
cl=pd.read_parquet(W+'claims_prepped.parquet'); cl['ly']=cl.loss_date.dt.year
lev=pd.read_parquet(W+'claims_level.parquet')[['claim_id','cost_adj']]
cl=cl[cl.status!='repudiated']
pol_c=cl.groupby(['policy_id','ly']).size().rename('n').reset_index()
cus_map=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['policy_id','customer_id']) for f in
    ['policies_motor','policies_buildings','policies_contents']]).drop_duplicates('policy_id')
cc=cl.merge(cus_map,on='policy_id').groupby(['customer_id','line','ly']).size().unstack('line',fill_value=0).add_prefix('cust_prev_').reset_index()
def build(df,ln):
    df=df.copy()
    df['ps']=pd.to_datetime(df.period_start); df['year']=df.ps.dt.year
    # DQ fix 1: some training rows carry exposure in days (365, 327...) and rounding variants; recompute from dates
    df['exposure']=((pd.to_datetime(df.period_end)-df.ps).dt.days+1)/365
    # DQ fix 2: construction_type has 'T'/'t' variants of 'timber' in training only
    if 'construction_type' in df: df['construction_type']=df.construction_type.replace({'T':'timber','t':'timber'})
    df['month_start']=df.ps.dt.month
    df=df.merge(cust,on='customer_id',how='left')
    df['cust_age']=(df.ps-df.date_of_birth).dt.days/365.25
    df['tenure_yrs']=df.year-df.customer_since_year
    df=df.merge(addr,on='address_id',how='left')
    df['prop_age']=df.year-df.year_built
    df=df.merge(wx,on=['address_id','year'],how='left')
    # prior-year claim history (unknown for 2021)
    h=pol_c.assign(year=pol_c.ly+1).drop(columns='ly').rename(columns={'n':'pol_prev_claims'})
    df=df.merge(h,on=['policy_id','year'],how='left')
    h2=cc.assign(year=cc.ly+1).drop(columns='ly')
    df=df.merge(h2,on=['customer_id','year'],how='left')
    hc=[c for c in df.columns if c.startswith('cust_prev_')]+['pol_prev_claims']
    df[hc]=df[hc].fillna(0)
    df.loc[df.year==2021,hc]=np.nan
    df=df.drop(columns=['ps','date_of_birth','period_start','period_end','customer_since_year','year_built'],errors='ignore')
    return df
for ln,f,t in [('motor','policies_motor','test_motor'),('buildings','policies_buildings','test_buildings'),('contents','policies_contents','test_contents')]:
    tr=pd.read_parquet(P+f+'.parquet')
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(lev,on='claim_id')
    tgt=j.groupby('row_id').cost_adj.sum()
    tr=build(tr,ln); tr['target']=tr.row_id.map(tgt).fillna(0.0)
    te=build(pd.read_parquet(P+t+'.parquet'),ln)
    tr.to_parquet(W+f'mf_train_{ln}.parquet'); te.to_parquet(W+f'mf_test_{ln}.parquet')
    print(ln,tr.shape,te.shape,'target pp',round(tr.target.sum()/tr.exposure.sum(),2),'null wx',tr.rainfall_mm.isna().mean(),te.rainfall_mm.isna().mean())
