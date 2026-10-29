"""Assemble Part 2 submission: per-line stacked blend of T/L/O, new-business uplift, per-line level multipliers."""
import sys, json, pandas as pd, numpy as np
W2='/home/claude/iaa-pricing/work_p2/'
NB_ADJ={'motor':1.23,'buildings':1.00,'contents':1.19}      # credibility-weighted: local evidence vs developed-market NB uplift, divided by what the blend already predicts
def build(S, out, nb_adj=NB_ADJ):
    st=json.load(open(W2+'stack_p2.json')); parts=[]
    for ln in ['motor','buildings','contents']:
        p=pd.read_parquet(W2+f'pred_{ln}.parquet'); o=pd.read_parquet(W2+f'oof_{ln}.parquet'); w=st[ln]['w']
        lvl=o.y.sum()/o.exposure.sum()                     # pooled 2023-24 emerging pure premium
        comps=[p[c].values*o.y.sum()/o[c].sum() for c in ['T','L','O']]   # each component level-matched on train
        b=sum(wi*c for wi,c in zip(w,comps))
        b=b*np.where(p.new_business.values,nb_adj[ln],1.0)
        b=b*S[ln]
        parts.append(pd.DataFrame({'row_id':p.row_id,'prediction':b}))
        print(f'{ln}: test pp {b.sum()/p.exposure.sum():.1f} vs train pooled {lvl:.1f} | NB share {p.new_business.mean():.2f} | level mult {S[ln]}')
    s=pd.concat(parts); ss=pd.read_parquet('/home/claude/data_p2/emerging/sample_submission.parquet')
    assert set(s.row_id)==set(ss.row_id) and s.row_id.is_unique and (s.prediction>=0).all() and np.isfinite(s.prediction).all()
    s.to_csv(out,index=False); print('wrote',out,len(s))
if __name__=='__main__':
    build(json.loads(sys.argv[1]),sys.argv[2])
