"""Segment-level solve. 1-D2 = k + sum_seg (b_s s_s^-0.5 + c_s s_s^0.5), c_s = kappa * sum(sqrt(mu_s)).
Unknowns k, kappa, b_NB, b_REN (4). Equations: v1, global x1.35, NB x1.3, REN x1.3 (4)."""
import pandas as pd, numpy as np
from scipy.optimize import fsolve
W2='/home/claude/iaa-pricing/work_p2/'
s=pd.read_csv(W2+'p2_submission_v1.csv',dtype={'row_id':str})
nb=pd.concat([pd.read_parquet(W2+f'pred_{ln}.parquet',columns=['row_id','new_business']) for ln in ['motor','buildings','contents']]).set_index('row_id').new_business.reindex(s.row_id).values
C={'NB':np.sqrt(s.prediction[nb]).sum(),'REN':np.sqrt(s.prediction[~nb]).sum()}
obs=[({'NB':1,'REN':1},0.13787),({'NB':1.35,'REN':1.35},0.14409),({'NB':1.3,'REN':1},0.14492),({'NB':1,'REN':1.3},0.13767)]
def eqs(x):
    k,kap,bN,bR=x; b={'NB':bN,'REN':bR}
    return [k+sum(b[g]*sc[g]**-0.5+kap*C[g]*sc[g]**0.5 for g in b)-(1-d) for sc,d in obs]
k,kap,bN,bR=fsolve(eqs,[-0.2,1e-5,0.3,0.3]); b={'NB':bN,'REN':bR}; c={g:kap*C[g] for g in C}
sopt={g:b[g]/c[g] for g in b}; D2opt=1-k-sum(2*np.sqrt(b[g]*c[g]) for g in b)
def D2(sc): return 1-k-sum(b[g]*sc[g]**-0.5+c[g]*sc[g]**0.5 for g in b)
print('residuals',np.round(eqs([k,kap,bN,bR]),6))
print('optimal multipliers vs v1:',{g:round(v,3) for g,v in sopt.items()},'-> predicted D2',round(D2opt,5))
for sN in [1.4,1.6,1.8,2.0,2.2]:
    print(f'  NB x{sN}: ' + '  '.join(f'REN x{sR}: {D2({"NB":sN,"REN":sR}):.4f}' for sR in [0.9,1.0,1.1]))
