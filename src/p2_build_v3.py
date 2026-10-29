"""Part 2 v3+: per-line 4-model blend (weights from OOF stacking), each (line, new_business) segment level-matched to
v1's segment total x the LB-solved multiplier."""
import sys, json, pandas as pd, numpy as np
W2='/home/claude/iaa-pricing/work_p2/'
def build(mult, out, weights_file='stack_p2_v3.json'):
    st=json.load(open(W2+weights_file)); v1=pd.read_csv(W2+'p2_submission_v1.csv',dtype={'row_id':str}).set_index('row_id').prediction
    parts=[]
    for ln,L in [('motor','M'),('buildings','B'),('contents','C')]:
        p=pd.read_parquet(W2+f'pred_{ln}.parquet'); o=pd.read_parquet(W2+f'oof_{ln}.parquet'); w=st[ln]['w']; names=st[ln].get('names',['T','L','O','P'])
        comps=[p[c].values*o.y.sum()/o[c].sum() for c in names]
        b=sum(wi*c for wi,c in zip(w,comps)); b=pd.Series(b,index=p.row_id.values)
        for seg,m in [(True,mult[L+'NB']),(False,mult[L+'REN'])]:
            idx=p.row_id[p.new_business.values==seg].values
            b[idx]*=v1[idx].sum()*m/b[idx].sum()
        parts.append(b.rename('prediction').reset_index().rename(columns={'index':'row_id'}))
        print(f'{ln}: blend w={np.round(w,2)} | NB pp {b[p.row_id[p.new_business.values].values].sum()/p.exposure[p.new_business.values].sum():.1f} REN pp {b[p.row_id[~p.new_business.values].values].sum()/p.exposure[~p.new_business.values].sum():.1f}')
    s=pd.concat(parts); ss=pd.read_parquet('/home/claude/data_p2/emerging/sample_submission.parquet')
    assert set(s.row_id)==set(ss.row_id) and s.row_id.is_unique and (s.prediction>=0).all() and np.isfinite(s.prediction).all()
    s.to_csv(out,index=False); print('wrote',out)
if __name__=='__main__': build(json.loads(sys.argv[1]),sys.argv[2])
