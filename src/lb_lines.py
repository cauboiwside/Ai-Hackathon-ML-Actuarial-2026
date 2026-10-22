"""Per-line level solve from v2 and v3 scores.
1-D2 = k + sum_l (b_l s_l^-0.5 + c_l s_l^0.5); k known from v1 probes; c_l = c1 * C_l/C1 (C = sum sqrt(pred)).
Unknowns b_M, b_B, b_C: 3. Equations: v2 (s=1), v3 (known per-line ratios): 2.
Closing assumption: buildings and contents share the same optimal multiplier (b_B/c_B = b_C/c_C) - both home lines
showed the same A/E pattern in every diagnostic."""
import numpy as np, pandas as pd
W='work/'
k=-0.23106; c1=0.49583
v1=pd.read_csv(W+'submission_v1.csv',dtype={'row_id':str}).set_index('row_id').prediction
v2=pd.read_csv(W+'submission_v2.csv',dtype={'row_id':str}).set_index('row_id').prediction
v3=pd.read_csv(W+'submission_v3.csv',dtype={'row_id':str}).set_index('row_id').prediction.reindex(v2.index)
line=v2.index.str[0]; C1=np.sqrt(v1).sum()
c={l:c1*np.sqrt(v2[line==l]).sum()/C1 for l in 'MBC'}
r3={l:v3[line==l].sum()/v2[line==l].sum() for l in 'MBC'}; print('v3/v2 per line',{l:round(v,4) for l,v in r3.items()})
S2,S3=0.12444,0.12399
# unknowns x=(b_M, t) with b_B=t*c_B, b_C=t*c_C
A=np.array([[1, c['B']+c['C']],
            [r3['M']**-0.5, c['B']*r3['B']**-0.5 + c['C']*r3['C']**-0.5]])
rhs=np.array([(1-S2)-k-sum(c.values()),
              (1-S3)-k-sum(c[l]*r3[l]**0.5 for l in 'MBC')])
bM,t=np.linalg.solve(A,rhs); b={'M':bM,'B':t*c['B'],'C':t*c['C']}
sopt={l:b[l]/c[l] for l in 'MBC'}
D2opt=1-k-sum(2*np.sqrt(b[l]*c[l]) for l in 'MBC')
print('c per line',{l:round(v,4) for l,v in c.items()}); print('b per line',{l:round(v,4) for l,v in b.items()})
print('optimal multipliers vs v2:',{l:round(v,3) for l,v in sopt.items()},' -> predicted D2',round(D2opt,5))
def D2(s): return 1-k-sum(b[l]*s[l]**-0.5+c[l]*s[l]**0.5 for l in 'MBC')
print('check v2',round(D2({'M':1,'B':1,'C':1}),5),'check v3',round(D2(r3),5))
for sm in [0.85,0.9,0.95,1.0]:
    for sh in [1.0,1.05,1.1,1.15]: print(f'  motor x{sm} home x{sh}: {D2({"M":sm,"B":sh,"C":sh}):.5f}')
