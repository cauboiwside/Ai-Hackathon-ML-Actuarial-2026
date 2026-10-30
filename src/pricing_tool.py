"""Pricing tool: risk premium -> technical premium using the expenses table.
Premium P solves: P = R*(1+cat) + F*exposure + P*(var + commission + tax + profit)
=> P = (R*(1+cat_if_property) + F*exposure) / (1 - var - commission - tax - profit)."""
import pandas as pd, numpy as np, sys
E='/home/claude/data_p2/emerging/'; W2='/home/claude/iaa-pricing/work_p2/'
def price(rows, market='emerging', pred_file=W2+'p2_submission_v8.csv'):
    ex=pd.read_parquet(E+'expenses.parquet').set_index('item')[market]
    sub=pd.read_csv(pred_file,dtype={'row_id':str}).set_index('row_id').prediction
    r=rows.copy(); r['risk_premium']=r.row_id.map(sub)
    comm=r.channel.map({'direct':ex['commission_direct'],'broker':ex['commission_broker'],'comparison_site':ex['commission_comparison_site']}).fillna(0)
    cat=np.where(r.line.isin(['buildings','contents']),ex['cat_reinsurance'],0.0)
    load=1-ex['variable_expense']-comm-ex['premium_tax']-ex['target_profit_margin']
    r['technical_premium']=(r.risk_premium*(1+cat)+ex['fixed_expense_per_policy']*r.exposure)/load
    r['loading_ratio']=r.technical_premium/r.risk_premium
    return r
if __name__=='__main__':
    ref=pd.read_parquet(E+'reference_policies.parquet'); t=price(ref)
    cols=['row_id','line','channel','new_business','exposure','written_premium','risk_premium','technical_premium','loading_ratio']
    t=t[cols].round({'risk_premium':1,'technical_premium':1,'loading_ratio':2}); print(t.to_string(index=False)); t.to_csv('/home/claude/iaa-pricing/docs/reference_policies_priced.csv',index=False)
    print('\nmedian written/technical:',round((t.written_premium/t.technical_premium).median(),2))
