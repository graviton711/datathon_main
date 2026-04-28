"""
Investigate Day-of-Week correction.
1. Compute historical DoW lift factors from training data (2019-2022)
2. Check if current model captures these lifts correctly
3. Simulate applying a multiplicative DoW correction
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year'] = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['dow']  = sales['Date'].dt.dayofweek  # 0=Mon, 6=Sun
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

post = sales[sales['year'] >= 2019].copy()

# 1. Historical DoW lift (vs weekly mean within same year-month block)
print("=== 1. DOW LIFT RELATIVE TO WEEKLY MEAN (2019-2022) ===")
# Compute weekly mean per year-month
ym_means = post.groupby(['year','month'])['Revenue'].transform('mean')
post['rel_lift'] = post['Revenue'] / (ym_means + 1e-6)

dow_profile = post.groupby('dow')['rel_lift'].agg(['mean','median','std'])
dow_profile.index = [dow_names[i] for i in dow_profile.index]
dow_profile.columns = ['mean_lift','median_lift','std']
print(dow_profile.round(4).to_string())

# 2. By year - is the pattern stable?
print("\n=== 2. DOW MEDIAN LIFT BY YEAR (stability check) ===")
for yr in [2019, 2020, 2021, 2022]:
    yr_data = post[post['year']==yr]
    ym_m = yr_data.groupby(['year','month'])['Revenue'].transform('mean')
    yr_data = yr_data.copy()
    yr_data['rel_lift'] = yr_data['Revenue'] / (ym_m + 1e-6)
    yr_dow = yr_data.groupby('dow')['rel_lift'].median()
    row = [f"{yr_dow.get(d, 0):.4f}" for d in range(7)]
    print(f"  {yr}: {' '.join(row)}")

# 3. Current model DoW profile (from submission vs monthly mean)
print("\n=== 3. CURRENT MODEL DOW PROFILE ===")
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
best['Date']  = pd.to_datetime(best['Date'])
curr['dow'] = curr['Date'].dt.dayofweek
curr['month'] = curr['Date'].dt.month
curr['year']  = curr['Date'].dt.year

ym_means_curr = curr.groupby(['year','month'])['Revenue'].transform('mean')
curr['rel_lift'] = curr['Revenue'] / (ym_means_curr + 1e-6)
curr_dow = curr.groupby('dow')['rel_lift'].median()

# Historical from training data (median)
hist_dow = post.groupby('dow')['rel_lift'].median()

print(f"\n{'DoW':>4}  {'Hist':>8}  {'Curr':>8}  {'Gap':>8}")
for d in range(7):
    hist_v = hist_dow.get(d, 1.0)
    curr_v = curr_dow.get(d, 1.0)
    gap = curr_v - hist_v
    print(f"{dow_names[d]:>4}  {hist_v:>8.4f}  {curr_v:>8.4f}  {gap:>+8.4f}")

# 4. Simulate: apply correction factor = hist / curr for each DoW
# Only if the pattern is consistent across years (std < 0.05)
print("\n=== 4. SIMULATION: DoW correction ===")

# Correction: multiply each day by (hist_lift / curr_lift) for that DoW
correction = {}
for d in range(7):
    hist_v = hist_dow.get(d, 1.0)
    curr_v = curr_dow.get(d, 1.0)
    correction[d] = hist_v / curr_v

print("Corrections per DoW:")
for d in range(7):
    print(f"  {dow_names[d]}: {correction[d]:.4f}x")

# Apply correction to submission
sim_rev = curr['Revenue'].copy()
for d in range(7):
    mask = curr['dow'] == d
    sim_rev[mask] *= correction[d]

sim_df = curr[['Date']].copy()
sim_df['Revenue'] = sim_rev
sim_df['COGS'] = curr['COGS']

merged = pd.merge(sim_df, best, on='Date', suffixes=('_c','_b'))
baseline_mae = (pd.merge(curr[['Date','Revenue']], best, on='Date', suffixes=('_c','_b'))['Revenue_c'] -
                pd.merge(curr[['Date','Revenue']], best, on='Date', suffixes=('_c','_b'))['Revenue_b']).abs().mean()
sim_mae = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
sim_bias = (merged['Revenue_c'] - merged['Revenue_b']).mean()

print(f"\nBaseline MAE vs best_624k: {baseline_mae:,.0f}")
print(f"DoW corrected MAE:          {sim_mae:,.0f}  (delta={sim_mae-baseline_mae:+,.0f})")
print(f"DoW corrected bias:         {sim_bias:,.0f}")

# 5. Stability: only use correction if std < 0.1 per DoW
print("\n=== 5. STABILITY CHECK (std across years) ===")
print("Only apply correction for DoW with std < 0.10")
stable_dow = []
for d in range(7):
    yr_lifts = []
    for yr in [2019, 2020, 2021, 2022]:
        yr_data = post[post['year']==yr]
        ym_m = yr_data.groupby(['year','month'])['Revenue'].transform('mean')
        yr_data = yr_data.copy()
        yr_data['rel_lift'] = yr_data['Revenue'] / (ym_m + 1e-6)
        v = yr_data[yr_data['dow']==d]['rel_lift'].median()
        yr_lifts.append(v)
    std_v = np.std(yr_lifts)
    mean_v = np.mean(yr_lifts)
    stable = std_v < 0.10
    if stable: stable_dow.append(d)
    print(f"  {dow_names[d]}: mean={mean_v:.4f}  std={std_v:.4f}  {'STABLE' if stable else 'UNSTABLE'}")

print(f"\nStable DoW for correction: {[dow_names[d] for d in stable_dow]}")

# Simulate with stable-only corrections
sim_rev2 = curr['Revenue'].copy()
for d in stable_dow:
    mask = curr['dow'] == d
    sim_rev2[mask] *= correction[d]

sim_df2 = curr[['Date']].copy()
sim_df2['Revenue'] = sim_rev2
sim_df2['COGS'] = curr['COGS']
merged2 = pd.merge(sim_df2, best, on='Date', suffixes=('_c','_b'))
sim_mae2 = (merged2['Revenue_c'] - merged2['Revenue_b']).abs().mean()
print(f"\nStable-only DoW corrected MAE: {sim_mae2:,.0f}  (delta={sim_mae2-baseline_mae:+,.0f})")
