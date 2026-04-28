import pandas as pd
import numpy as np

best = pd.read_csv('data/best_submit/best_624k.csv')
curr = pd.read_csv('submissions/submission.csv')

best['Date'] = pd.to_datetime(best['Date'])
curr['Date'] = pd.to_datetime(curr['Date'])

merged = pd.merge(curr, best, on='Date', suffixes=('_curr', '_best'))
merged['year'] = merged['Date'].dt.year
merged['month'] = merged['Date'].dt.month
merged['dow'] = merged['Date'].dt.dayofweek  # 0=Mon, 6=Sun

# 1. Global stats
print('=== GLOBAL BIAS ===')
print(f"Current mean Rev : {merged['Revenue_curr'].mean():,.0f}")
print(f"Best mean Rev    : {merged['Revenue_best'].mean():,.0f}")
print(f"Ratio curr/best  : {merged['Revenue_curr'].mean() / merged['Revenue_best'].mean():.4f}")
print(f"Current mean COGS: {merged['COGS_curr'].mean():,.0f}")
print(f"Best mean COGS   : {merged['COGS_best'].mean():,.0f}")

# 2. MAE breakdown by year
print('\n=== BIAS BY YEAR ===')
for yr in [2023, 2024]:
    m = merged[merged['year'] == yr]
    mae_rev  = (m['Revenue_curr'] - m['Revenue_best']).abs().mean()
    bias_rev = (m['Revenue_curr'] - m['Revenue_best']).mean()
    mae_cogs = (m['COGS_curr'] - m['COGS_best']).abs().mean()
    print(f"{yr}: Rev MAE={mae_rev:,.0f}  Rev Bias={bias_rev:,.0f}  COGS MAE={mae_cogs:,.0f}")

# 3. Monthly ratio
print('\n=== MONTHLY RATIO (curr_mean / best_mean) ===')
monthly = merged.groupby(['year','month']).agg(
    curr_rev=('Revenue_curr','mean'),
    best_rev=('Revenue_best','mean'),
    curr_cogs=('COGS_curr','mean'),
    best_cogs=('COGS_best','mean'),
).reset_index()
monthly['rev_ratio'] = monthly['curr_rev'] / monthly['best_rev']
monthly['rev_diff']  = monthly['curr_rev'] - monthly['best_rev']
print(monthly[['year','month','best_rev','curr_rev','rev_ratio','rev_diff']].to_string(index=False))

# 4. Day-of-Week profile
print('\n=== DAY-OF-WEEK: curr vs best mean Rev ===')
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow = merged.groupby('dow').agg(
    curr_rev=('Revenue_curr','mean'),
    best_rev=('Revenue_best','mean'),
).reset_index()
dow['ratio'] = dow['curr_rev'] / dow['best_rev']
dow['dow_name'] = dow['dow'].map(lambda x: dow_names[x])
print(dow[['dow_name','best_rev','curr_rev','ratio']].to_string(index=False))

# 5. Day-of-month profile
print('\n=== TOP 10 HIGHEST MAE DAYS-OF-MONTH ===')
dom = merged.copy()
dom['dom'] = dom['Date'].dt.day
dom_agg = dom.groupby('dom').agg(
    mae_rev=('Revenue_curr', lambda x: (x - dom.loc[x.index,'Revenue_best']).abs().mean()),
    bias_rev=('Revenue_curr', lambda x: (x - dom.loc[x.index,'Revenue_best']).mean()),
).reset_index()
print(dom_agg.sort_values('mae_rev', ascending=False).head(10).to_string(index=False))
