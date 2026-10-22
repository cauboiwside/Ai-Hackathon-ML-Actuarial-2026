import sys; sys.path.insert(0,'/home/claude/iaa-pricing/src')
import numpy as np, pandas as pd
from common import *
o=pd.concat([pd.read_parquet(W+f'oof_{ln}.parquet').assign(line=ln) for ln in ['motor','buildings','contents']])
print('combined OOF D2 (adjusted target):',round(d2(o.y.values,o.oof.values),4))
for s in [0.7,0.8,0.9,1.0,1.1,1.2]:
    print(' scale',s,'D2',round(d2(o.y.values,o.oof.values*s),4))
# ranking quality against RAW observed cost (unadjusted), 2024 only and all years
raw=pd.concat([pd.read_parquet(W+f'train_{ln}.parquet',columns=['row_id','cost']) for ln in ['motor','buildings','contents']])
o=o.merge(raw,on='row_id')
for yr in [None,2024]:
    q=o if yr is None else o[o.year==yr]
    best=max((d2(q.cost.values,q.oof.values*s),s) for s in np.arange(0.5,1.3,0.05))
    print('vs RAW cost',yr or 'all','D2 at s=1:',round(d2(q.cost.values,q.oof.values),4),'best scale',round(best[1],2),'D2',round(best[0],4))
print(o.groupby('line').apply(lambda g: pd.Series({'adj_pp':g.y.sum()/g.exposure.sum(),'raw_pp':g.cost.sum()/g.exposure.sum(),'oof_pp':g.oof.sum()/g.exposure.sum()}),include_groups=False).round(1))
