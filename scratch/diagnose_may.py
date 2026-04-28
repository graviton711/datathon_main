"""
Diagnose May 2023 underestimation:
- Is it a seasonal suppression (same as Sep)? → floor would fix it
- Or is it a different mechanism (model not capturing peak)? → floor won't trigger
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales = sales.sort_values('Date').reset_index(drop=True)
sales['trail60'] = sales['Revenue'].shift(1).rolling(60, min_periods=30).mean()
sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month

# 1. May/trail60 ratio historically (post-2019)
print("=== 1. MAY_MEAN / TRAIL60 AT MAY-START (2019-2022) ===")
post = sales[sales['year'] >= 2019]
may_ratios = []
for yr in [2019, 2020, 2021, 2022]:
    may_rows = post[(post['year']==yr) & (post['month']==5)]
    if may_rows.empty: continue
    trail_val = post.loc[may_rows.index[0], 'trail60']
    mo_mean   = may_rows['Revenue'].mean()
    ratio = mo_mean / trail_val
    may_ratios.append(ratio)
    print(f"  {yr}: mo_mean={mo_mean:,.0f}  trail60={trail_val:,.0f}  ratio={ratio:.4f}")

print(f"  -> min={min(may_ratios):.4f}  median={np.median(may_ratios):.4f}  max={max(may_ratios):.4f}")

# 2. What is the trail60 at May 1, 2023 in the current submission?
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
best['Date']  = pd.to_datetime(best['Date'])
curr = curr.sort_values('Date').reset_index(drop=True)

may_2023_idx = curr[curr['Date'].dt.month == 5].index[0]
trail_at_may = curr.loc[max(0, may_2023_idx-60):may_2023_idx-1, 'Revenue'].mean()
curr_may_mean = curr[curr['Date'].dt.month == 5]['Revenue'].mean()
curr_may_ratio = curr_may_mean / trail_at_may
best_may_mean = best[best['Date'].dt.month == 5]['Revenue'].mean()

print(f"\n=== 2. CURRENT SUBMISSION: May 2023 situation ===")
print(f"  trail60 at May 1, 2023: {trail_at_may:,.0f}")
print(f"  Current May mean:       {curr_may_mean:,.0f}")
print(f"  Current May/trail60:    {curr_may_ratio:.4f}")
print(f"  Best May mean:          {best_may_mean:,.0f}")
print(f"  Best May/trail60:       {best_may_mean/trail_at_may:.4f}")
print(f"  Historical median:      {np.median(may_ratios):.4f}")
print()
if curr_may_ratio > np.median(may_ratios):
    print("  >>> FLOOR WOULD NOT TRIGGER: current prediction ALREADY ABOVE historical median.")
    print("  >>> May underestimation is NOT a seasonal suppression issue.")
else:
    print("  >>> FLOOR WOULD TRIGGER: current prediction below historical median.")
    print("  >>> May underestimation IS a seasonal suppression issue — same fix as Sep.")

# 3. What floor alpha would need to be to reach best_may?
required_alpha = best_may_mean / trail_at_may
print(f"\n  Alpha needed to reach best_624k May: {required_alpha:.4f}")
print(f"  This is {'above' if required_alpha > max(may_ratios) else 'within'} historical range (max={max(may_ratios):.4f})")

# 4. Diagnosis: what is actually causing May underprediction?
print("\n=== 3. ROOT CAUSE ANALYSIS ===")
# Check if it's a blended momentum issue or event lift issue
# Compare Mar/Apr/May prediction sequence
for mo in [3, 4, 5, 6]:
    mo_curr = curr[curr['Date'].dt.month == mo]['Revenue'].mean()
    mo_best = best[best['Date'].dt.month == mo]['Revenue'].mean()
    print(f"  Month {mo:2d}: curr={mo_curr:,.0f}  best={mo_best:,.0f}  ratio={mo_curr/mo_best:.4f}")

# Check: is May being pulled down because Oct-Dec momentum is too low?
# (since prev_q4_momentum affects the whole year)
print("\n=== 4. TRAINING DATA: HISTORICAL Apr-May TRANSITION ===")
for yr in [2019, 2020, 2021, 2022]:
    apr_rows = post[(post['year']==yr) & (post['month']==4)]
    may_rows = post[(post['year']==yr) & (post['month']==5)]
    if apr_rows.empty or may_rows.empty: continue
    apr_mean = apr_rows['Revenue'].mean()
    may_mean = may_rows['Revenue'].mean()
    print(f"  {yr}: Apr={apr_mean:,.0f}  May={may_mean:,.0f}  May/Apr={may_mean/apr_mean:.4f}")
