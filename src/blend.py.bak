"""Blend v2 with partner's model. Partner predictions are level-matched to v2 per line first
(v2's total level is LB-near-optimal), then mixed arithmetically with weight w on the partner."""
import sys, pandas as pd, numpy as np
W='work/'; w=float(sys.argv[1]); out=sys.argv[2]; tot_scale=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
ours=pd.read_csv(W+'submission_v2.csv',dtype={'row_id':str}).set_index('row_id').prediction
his=sum(pd.read_csv(f'/home/claude/partner/{f}.csv',dtype={'row_id':str}).set_index('row_id').prediction.reindex(ours.index) for f in ['risk_v1','risk_v2'])/2
line=ours.index.str[0]
for l in 'MBC':
    m=line==l; his[m]*=ours[m].sum()/his[m].sum()
blend=((1-w)*ours+w*his)*tot_scale
print(f'w={w} tot_scale={tot_scale} | blend/ours total {blend.sum()/ours.sum():.3f} | corr(blend,ours) {blend.corr(ours):.4f} | min {blend.min():.3f} max {blend.max():.1f}')
blend.rename('prediction').reset_index().to_csv(out,index=False)
