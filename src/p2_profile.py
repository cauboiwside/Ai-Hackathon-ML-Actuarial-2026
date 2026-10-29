import pandas as pd, numpy as np
E='/home/claude/data_p2/emerging/'; D='/home/claude/data_raw/public/'
pd.set_option('display.width',250); pd.set_option('display.max_columns',60)
for f in ['customers','addresses','weather','policies_motor','policies_buildings','policies_contents','claims','test_motor','test_buildings','test_contents','sample_submission','expenses','reference_policies']:
    d=pd.read_parquet(E+f+'.parquet'); print(f'\n=== {f} {d.shape}')
    try:
        dd=pd.read_parquet(D+f+'.parquet'); extra=set(d.columns)-set(dd.columns); miss=set(dd.columns)-set(d.columns)
        print('   dev shape',dd.shape,'| new cols',sorted(extra),'| missing cols',sorted(miss))
    except Exception: print('   (no developed counterpart)'); print(d.head(5).to_string()); print(d.describe(include='all').T.head(30).to_string())
    if f in ('expenses','reference_policies'): continue
    na=d.isna().mean(); na=na[na>0].round(3)
    if len(na): print('   NA share:',na.to_dict())
