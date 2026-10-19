"""Step 2: peril-level IBNR, large-loss cap, case development, trend -> per-claim on-level factor."""
import pandas as pd, numpy as np, json
W='/home/claude/iaa-pricing/work/'; VAL=pd.Timestamp('2024-12-31')
c=pd.read_parquet(W+'claims_prepped.parquet'); c['ly']=c.loss_date.dt.year
CAP={'motor':250_000,'buildings':300_000,'contents':60_000}
c['cap']=c.line.map(CAP); c['cost_c']=c[['cost','cap']].min(axis=1)
# peril-level report-lag CDF from 2021 losses (>=3y observed)
ref=c[c.ly==2021]
c['Fp']=np.nan
for (l,p),g in c.groupby(['line','peril']):
    lags=np.sort(ref[(ref.line==l)&(ref.peril==p)].lag.values)
    c.loc[g.index,'Fp']=np.searchsorted(lags,g.age.values,side='right')/len(lags)
c['Fp']=c.Fp.clip(lower=0.2)
c['w']=1/c.Fp
nr=c[(c.status!='repudiated')&(c.ly>=2021)]
# severity per line/peril/year (capped)
sev=nr.groupby(['line','peril','ly']).cost_c.mean().unstack()
openp=nr.assign(o=nr.status=='open').groupby(['line','peril','ly']).o.mean().unstack()
# severity trend per line: log-linear on years, perils with low open share, weighted by count
tr={}
for ln in CAP:
    rows=[]
    for (l,p),r in sev.loc[[ln]].iterrows():
        if openp.loc[(l,p)].max()<0.10:
            n=nr[(nr.line==l)&(nr.peril==p)].groupby('ly').size()
            for y in r.index: rows.append((y,np.log(r[y]/r.mean()),n[y]))
    d=np.array(rows); b=np.polyfit(d[:,0],d[:,1],1,w=np.sqrt(d[:,2]))[0]; tr[ln]=float(np.exp(b)-1)
print('severity trend', {k:round(v,4) for k,v in tr.items()})
# development: for perils with open share >=10% in a year, set severity to trended mean of years with open<10%
dev={}
for (l,p),r in sev.iterrows():
    # long-tail perils: open share is structurally high, treat 2021-22 (2+ yrs old) as mature
    mature=[2021,2022] if p in ('bodily_injury','subsidence') else [y for y in r.index if openp.loc[(l,p),y]<0.10]
    t=1+tr[l]
    base=np.mean([r[y]*t**(2024-y) for y in mature]) if mature else None
    for y in r.index:
        f=1.0
        if y not in mature and base:
            f=max(1.0,base/t**(2024-y)/r[y])
        dev[(l,p,y)]=f
dv=pd.Series(dev).unstack().round(3); print('case-development factors\n',dv[dv.max(axis=1)>1])
c['dev']=[dev.get((l,p,y),1.0) for l,p,y in zip(c.line,c.peril,c.ly)]
c['trend']=(1+c.line.map(tr))**(2025-c.ly)
c['cost_adj']=np.where(c.status=='repudiated',0,c.cost_c*c.w*c.dev*c.trend)
# large-loss load: excess over cap / capped, mature years
x=c[(c.ly.between(2021,2023))&(c.status!='repudiated')]
ll={ln:float(x[x.line==ln].cost.sum()/x[x.line==ln].cost_c.sum()) for ln in CAP}
print('large loss load',ll)
c[['claim_id','line','peril','ly','status','cost','cost_c','w','dev','trend','cost_adj']].to_parquet(W+'claims_level.parquet')
json.dump({'cap':CAP,'trend':tr,'ll':ll},open(W+'level_params.json','w'),indent=1)
# summary: frequency (IBNR) and pure premium at 2025 level by year
for ln in CAP:
    p=pd.read_parquet(W+f'train_{ln}.parquet',columns=['row_id','year','exposure'])
    j=pd.read_parquet(W+f'claimrows_{ln}.parquet',columns=['claim_id','row_id']).merge(c[['claim_id','cost_adj','w','status']],on='claim_id')
    j=j.merge(p[['row_id','year']],on='row_id')
    e=p.groupby('year').exposure.sum()
    s=pd.DataFrame({'freq_ult':j[j.status!='repudiated'].groupby('year').w.sum()/e,'pp2025':j.groupby('year').cost_adj.sum()/e})
    print(ln); print(s.round(4))
    q=j[j.status!='repudiated'].merge(c[['claim_id','peril']],on='claim_id').groupby(['peril','year']).w.sum().unstack()
    print((q/e.values).round(5))
