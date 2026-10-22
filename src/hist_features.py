"""History features: claims in prior 1 and 2 years at policy, customer and address level (NaN where not observable),
peril-specific address history for home lines, weather anomalies vs address mean."""
import pandas as pd, numpy as np
P='/home/claude/data_raw/public/'; W='/home/claude/iaa-pricing/work/'
def build_history():
    cp=pd.read_parquet(W+'claims_prepped.parquet'); cp=cp[cp.status!='repudiated']; cp['ly']=cp.loss_date.dt.year
    cl=pd.read_parquet(W+'claims_level.parquet')[['claim_id','cost_c']]; cp=cp.merge(cl,on='claim_id')
    pols=pd.concat([pd.read_parquet(P+f+'.parquet',columns=['policy_id','customer_id','address_id']) for f in ['policies_motor','policies_buildings','policies_contents']]).drop_duplicates('policy_id')
    cp=cp.merge(pols,on='policy_id',how='left')
    out={}
    for key in ['policy_id','customer_id','address_id']:
        g=cp.groupby([key,'ly']).agg(n=('claim_id','size'),c=('cost_c','sum')).reset_index()
        out[key]=g
    # address peril flags
    ap=cp[cp.peril.isin(['flood','escape_of_water','subsidence','theft'])].groupby(['address_id','ly','peril']).size().unstack('peril',fill_value=0).reset_index()
    out['addr_peril']=ap
    wx=pd.read_parquet(P+'weather.parquet'); wcols=[c for c in wx.columns if c not in ('address_id','year')]
    m=wx[wx.year<=2024].groupby('address_id')[wcols].mean().add_suffix('_amean')
    out['wx_mean']=m
    return out
def add_history(df,H):
    df=df.copy(); yr=df.year
    for key,pre in [('policy_id','pol'),('customer_id','cust'),('address_id','addr')]:
        g=H[key]
        for k in [1,2]:
            # claims in the k prior years
            s=g[g.ly>=2021].copy()
            for lag in range(1,k+1):
                t=s.assign(year=s.ly+lag).groupby([key,'year']).agg(n=('n','sum'),c=('c','sum')).reset_index()
                if lag==1: acc=t
                else: acc=pd.concat([acc,t]).groupby([key,'year']).sum().reset_index()
            df=df.merge(acc.rename(columns={'n':f'{pre}_n_prev{k}','c':f'{pre}_c_prev{k}'}),on=[key,'year'],how='left')
            obs=(yr-k)>=2021   # window fully observable?
            for c in [f'{pre}_n_prev{k}',f'{pre}_c_prev{k}']: df[c]=np.where(obs,df[c].fillna(0),np.nan)
    ap=H['addr_peril']; acc=None
    for lag in range(1,4):
        t=ap.assign(year=ap.ly+lag).drop(columns='ly')
        acc=t if acc is None else pd.concat([acc,t])
    acc=acc.groupby(['address_id','year']).sum().reset_index().drop(columns=[c for c in ['ly'] if c in acc])
    acc=acc.rename(columns={c:f'addr_{c}_prev3' for c in acc.columns if c not in ('address_id','year')})
    df=df.merge(acc,on=['address_id','year'],how='left')
    for c in [c for c in df.columns if c.endswith('_prev3')]: df[c]=np.where(yr>=2022,df[c].fillna(0),np.nan)
    df=df.merge(H['wx_mean'],on='address_id',how='left')
    for c in [c for c in H['wx_mean'].columns]:
        b=c.replace('_amean',''); df[b+'_anom']=df[b]-df[c]
    df=df.drop(columns=list(H['wx_mean'].columns))
    return df
if __name__=='__main__':
    H=build_history()
    for ln in ['motor','buildings','contents']:
        for s in ['train','test']:
            d=pd.read_parquet(W+f'mf_{s}_{ln}.parquet'); d2=add_history(d,H); d2.to_parquet(W+f'mfh_{s}_{ln}.parquet')
            print(ln,s,d.shape,'->',d2.shape)
