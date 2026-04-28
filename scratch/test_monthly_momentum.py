"""
Check: Does per-month YoY growth rate vary significantly vs global growth?
If monthly growth is CONSISTENT across years (low std) and DIFFERENT from
the global 1.277x, it's a genuine signal to implement.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month

# Monthly mean revenue by year (post-2019)
post = sales[sales['year'] >= 2019]
monthly_means = post.groupby(['year', 'month'])['Revenue'].mean().unstack('month')
print("=== MONTHLY MEAN REVENUE BY YEAR ===")
print(monthly_means.round(0).to_string())

# Per-month YoY growth rates
print("\n=== PER-MONTH YOY GROWTH RATES ===")
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_growth = {}
print(f"{'Month':>5}  {'2020/19':>8}  {'2021/20':>8}  {'2022/21':>8}  {'Mean':>8}  {'Std':>8}  {'vs_global':>10}")

global_mean_growth = 1.277  # pipeline's base_momentum
for m in range(1, 13):
    rates = []
    for yr_pair in [(2019,2020), (2020,2021), (2021,2022)]:
        yr_prev, yr_curr = yr_pair
        if yr_prev in monthly_means.index and yr_curr in monthly_means.index:
            prev = monthly_means.loc[yr_prev, m]
            curr = monthly_means.loc[yr_curr, m]
            if prev > 0 and not np.isnan(prev) and not np.isnan(curr):
                rates.append(curr / prev)
    if rates:
        mean_g = np.mean(rates)
        std_g  = np.std(rates)
        vs_global = mean_g / global_mean_growth
        month_growth[m] = mean_g
        print(f"{MONTH_NAMES[m-1]:>5}  {rates[0]:>8.4f}  {rates[1]:>8.4f}  {rates[2]:>8.4f}  {mean_g:>8.4f}  {std_g:>8.4f}  {vs_global:>+10.4f}")

print(f"\n  Global base_momentum: {global_mean_growth:.4f}x")

# Simulate: apply per-month momentum to current submission
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
best['Date']  = pd.to_datetime(best['Date'])
curr['month'] = curr['Date'].dt.month
curr['year']  = curr['Date'].dt.year

print("\n=== SIMULATION: per-month momentum adjustment ===")

# Scale approach: adjust each month's 2023 predictions by (per_month_growth / global_growth)
# Only for 2023 (Y+1) where momentum applies directly
correction_adj = curr['Revenue'].values.copy()
for m in range(1, 13):
    if m not in month_growth:
        continue
    adj = month_growth[m] / global_mean_growth
    adj = np.clip(adj, 0.85, 1.15)  # don't overcorrect
    mask = (curr['year'] == 2023) & (curr['month'] == m)
    correction_adj[mask] *= adj

# Also apply compounded for 2024 (Y+2)
# For 2024, use same per-month growth again (compounded)
for m in range(1, 13):
    if m not in month_growth:
        continue
    adj_2024 = (month_growth[m] / global_mean_growth) ** 2
    adj_2024 = np.clip(adj_2024, 0.75, 1.25)
    mask = (curr['year'] == 2024) & (curr['month'] == m)
    correction_adj[mask] *= adj_2024

merged = pd.merge(
    pd.DataFrame({'Date': curr['Date'], 'Revenue': correction_adj}),
    best[['Date','Revenue']], on='Date', suffixes=('_c','_b')
)
baseline_mae = (pd.merge(curr[['Date','Revenue']], best, on='Date', suffixes=('_c','_b'))
                ['Revenue_c'] - pd.merge(curr[['Date','Revenue']], best, on='Date', suffixes=('_c','_b'))['Revenue_b']).abs().mean()
sim_mae  = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
sim_b23  = (merged[merged['Date'].dt.year==2023]['Revenue_c'] -
            merged[merged['Date'].dt.year==2023]['Revenue_b']).mean()
sim_b24  = (merged[merged['Date'].dt.year==2024]['Revenue_c'] -
            merged[merged['Date'].dt.year==2024]['Revenue_b']).mean()

print(f"  Baseline MAE:  {baseline_mae:,.0f}")
print(f"  Adjusted MAE:  {sim_mae:,.0f}  (delta={sim_mae-baseline_mae:+,.0f})")
print(f"  Bias 2023: {sim_b23:+,.0f}")
print(f"  Bias 2024: {sim_b24:+,.0f}")

# Show per-month effect
print("\n=== PER-MONTH ADJUSTMENT FACTORS ===")
print(f"{'Month':>5}  {'Adj':>7}  {'May/Oct note':>25}")
for m in range(1, 13):
    if m in month_growth:
        adj = np.clip(month_growth[m] / global_mean_growth, 0.85, 1.15)
        print(f"{MONTH_NAMES[m-1]:>5}  {adj:>7.4f}x")
