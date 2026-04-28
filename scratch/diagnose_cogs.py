"""
Investigate COGS/Revenue ratio patterns in training data.
Goal: find an honest way to correct the 4% global underestimation.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['ratio'] = sales['COGS'] / (sales['Revenue'] + 1e-6)

# 1. Annual COGS ratio trend
print("=== 1. ANNUAL COGS/Rev RATIO (all years) ===")
annual = sales.groupby('year')['ratio'].median()
print(annual.round(4).to_string())

# 2. Post-2019 only
print("\n=== 2. POST-2019 ANNUAL RATIO ===")
post = sales[sales['year'] >= 2019]
print(post.groupby('year')['ratio'].median().round(4).to_string())

# 3. Monthly ratio profile: all vs post-2019
print("\n=== 3. MONTHLY RATIO PROFILE ===")
print(f"{'Month':>6}  {'All years':>10}  {'Post-2019':>10}  {'Diff':>8}")
full_monthly = sales.groupby('month')['ratio'].median()
post_monthly = post.groupby('month')['ratio'].median()
for mo in range(1, 13):
    diff = post_monthly.get(mo, 0) - full_monthly.get(mo, 0)
    print(f"{mo:>6}  {full_monthly.get(mo,0):>10.4f}  {post_monthly.get(mo,0):>10.4f}  {diff:>+8.4f}")

# 4. Current pipeline cogs_profile (what the model uses)
print("\n=== 4. CURRENT MODEL cogs_monthly_profile ===")
cogs_profile = sales.groupby('month').apply(
    lambda x: (x['COGS'] / (x['Revenue'] + 1e-6)).median()
).to_dict()
print({k: round(v, 4) for k, v in sorted(cogs_profile.items())})

# 5. Post-2019 version
post_cogs_profile = post.groupby('month').apply(
    lambda x: (x['COGS'] / (x['Revenue'] + 1e-6)).median()
).to_dict()
print("\n=== 5. POST-2019 cogs_monthly_profile ===")
print({k: round(v, 4) for k, v in sorted(post_cogs_profile.items())})

# 6. Is there a linear trend in annual ratio? Project to 2023-2024
print("\n=== 6. LINEAR TREND in COGS ratio (2019-2022) ===")
post4 = sales[sales['year'].isin([2019,2020,2021,2022])]
ann4 = post4.groupby('year')['ratio'].median().reset_index()
x = ann4['year'].values
y = ann4['ratio'].values
coeffs = np.polyfit(x, y, 1)
slope, intercept = coeffs
print(f"  Slope: {slope:.6f} per year")
print(f"  Intercept: {intercept:.4f}")
for yr in [2019,2020,2021,2022,2023,2024]:
    proj = slope * yr + intercept
    actual = ann4.loc[ann4['year']==yr, 'ratio'].values
    actual_str = f"actual={actual[0]:.4f}" if len(actual) else "projected"
    print(f"  {yr}: {proj:.4f} ({actual_str})")

# 7. Simulate: if we used post-2019 COGS profile, what would current COGS look like?
print("\n=== 7. SIMULATION: post-2019 profile vs current ===")
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
best['Date']  = pd.to_datetime(best['Date'])

curr['month'] = curr['Date'].dt.month

# Map post-2019 ratio to predictions
curr['post19_ratio'] = curr['month'].map(post_cogs_profile)
curr['cogs_post19']  = curr['Revenue'] * curr['post19_ratio']

merged = pd.merge(curr, best, on='Date', suffixes=('_c','_b'))
cogs_mae_current  = (merged['COGS_c']       - merged['COGS_b']).abs().mean()
cogs_mae_post19   = (curr['cogs_post19']     - merged['COGS_b']).abs().mean()
cogs_bias_current = (merged['COGS_c']        - merged['COGS_b']).mean()
cogs_bias_post19  = (curr['cogs_post19']     - merged['COGS_b']).mean()

print(f"  Current COGS MAE vs best:   {cogs_mae_current:,.0f}  Bias: {cogs_bias_current:,.0f}")
print(f"  Post-2019 COGS MAE vs best: {cogs_mae_post19:,.0f}  Bias: {cogs_bias_post19:,.0f}")
