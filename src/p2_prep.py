"""Part 2 data prep: emerging-market frames with DQ fixes, claims matched, capped target; plus developed-market
frames restricted to the COMMON feature set (ages banded, missing features dropped) for the transfer model."""
import pandas as pd, numpy as np, os
E='/home/claude/data_p2/emerging/'; D='/home/claude/data_raw/public/'; W2='/home/claude/iaa-pricing/work_p2/'; os.makedirs(W2,exist_ok=True)
CAP={'motor':250_000,'buildings':300_000,'contents':60_000}
def band(age): return pd.cut(age,[0,24,34,49,64,200],labels=['17-24','25-34','35-49','50-64','65+']).astype(str)
def fix_common(df):
    df=df.copy(); df['ps']=pd.to_datetime(df.period_start); df['year']=df.ps.dt.year
    df['exposure']=((pd.to_datetime(df.period_end)-df.ps).dt.days+1)/365
    df['month_start']=df.ps.dt.month
    if 'construction_type' in df:
        df['construction_type']=df.construction_type.astype(str).str.strip().str.lower().replace({'w':'timber','t':'timber','brk':'brick','stn':'stone','wod':'timber','mix':'non_standard','nan':np.nan})
    return df
def load_market(root, emerging):
    cust=pd.read_parquet(root+'customers.parquet'); addr=pd.read_parquet(root+'addresses.parquet'); wx=pd.read_parquet(root+'weather.parquet')
    cust['occupation']=cust.occupation.astype(str).str.strip().replace({'nan':np.nan,'None':np.nan})
    if 'date_of_birth' in cust: cust['dob']=pd.to_datetime(cust.date_of_birth)
    cl=pd.read_parquet(root+'claims.parquet'); cl['loss_date']=pd.to_datetime(cl.loss_date)
    cl['cost']=np.where(cl.status=='repudiated',0.0,(cl.incurred_to_date-cl.recoveries).clip(lower=0)); cl['cost_c']=np.minimum(cl.cost,cl.line.map(CAP))
    out={}
    for ln,f,t in [('motor','policies_motor','test_motor'),('buildings','policies_buildings','test_buildings'),('contents','policies_contents','test_contents')]:
        tr=fix_common(pd.read_parquet(root+f+'.parquet')); te=fix_common(pd.read_parquet(root+t+'.parquet'))
        # claims -> rows
        c=cl[cl.line==ln][['claim_id','policy_id','loss_date','cost_c','peril']].merge(tr[['row_id','policy_id','ps','period_end']],on='policy_id')
        c=c[(c.loss_date>=c.ps)&(c.loss_date<=pd.to_datetime(c.period_end))].drop_duplicates('claim_id')
        tr['target']=tr.row_id.map(c.groupby('row_id').cost_c.sum()).fillna(0.0); tr['n_claims']=tr.row_id.map(c.groupby('row_id').size()).fillna(0)
        c[['claim_id','row_id','cost_c','peril']].to_parquet(W2+f'claimrows_{"em" if emerging else "dev"}_{ln}.parquet')
        for df in (tr,te):
            df.drop(columns=[k for k in ['surname'] if k in df],inplace=True,errors='ignore')
            m=df.merge(cust.drop(columns=[k for k in ['date_of_birth','surname'] if k in cust]),on='customer_id',how='left')
            if 'dob' in m: m['cust_age']=(m.ps-m.dob).dt.days/365.25; m.drop(columns='dob',inplace=True)
            m['tenure_yrs']=m.year-m.customer_since_year
            m=m.merge(addr,on='address_id',how='left'); m['prop_age']=m.year-m.year_built
            m=m.merge(wx,on=['address_id','year'],how='left')
            # harmonise ages to bands (developed has exact ages)
            if ln=='motor':
                if 'driver_age' in m: m['driver_age_band']=band(m.driver_age)
            else:
                if 'policyholder_age' in m: m['policyholder_age_band']=band(m.policyholder_age)
            if 'cust_age' in m: m['cust_age_band']=band(m.cust_age)
            m.drop(columns=[k for k in ['ps','period_start','period_end','customer_since_year','year_built','driver_age','policyholder_age','cust_age',
                            'annual_mileage','telematics_opt_in','telematics_score','credit_band','gender','renewal_decision'] if k in m],inplace=True)
            df.attrs['out']=m
        out[ln]=(tr.attrs['out'],te.attrs['out'])
    return out
em=load_market(E,True); dv=load_market(D,False)
for ln in em:
    a,b=em[ln]; c,d=dv[ln]
    common=[k for k in a.columns if k in c.columns]; 
    print(ln,'emerging train',a.shape,'test',b.shape,'| developed common-feature train',c[common].shape,'| em-only cols',sorted(set(a.columns)-set(c.columns)),'| dev-only',sorted(set(c.columns)-set(a.columns)))
    a.to_parquet(W2+f'em_train_{ln}.parquet'); b.to_parquet(W2+f'em_test_{ln}.parquet'); c[common].to_parquet(W2+f'dev_train_{ln}.parquet')
    e=a.groupby(['year','new_business']).apply(lambda g: pd.Series({'expo':g.exposure.sum(),'freq':g.n_claims.sum()/g.exposure.sum(),'pp':g.target.sum()/g.exposure.sum()}),include_groups=False).round(3)
    print(e); print('  developed pooled freq',round(c.n_claims.sum()/c.exposure.sum(),4),'pp',round(c.target.sum()/c.exposure.sum(),1),'| by new_business:',c.groupby('new_business').apply(lambda g: round(g.target.sum()/g.exposure.sum(),1),include_groups=False).to_dict())
