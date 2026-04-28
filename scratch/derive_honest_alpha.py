"""
Derive floor alpha from training data (2019-2022).
Question: historically, what is Sep/Oct mean as a fraction of the
trailing 60-day mean just before them?

This gives a data-driven alpha without any knowledge of 2023 ground truth.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales = sales.sort_values('Date').reset_index(drop=True)

# Add trailing 60-day mean column (rolling mean of prev 60 days)
sales['trail60'] = sales['Revenue'].shift(1).rolling(60, min_periods=30).mean()

sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month

# For Sep and Oct each year: ratio = month_mean / trailing60_at_month_start
print("=== RATIO: Month mean / Trailing-60d mean at month start (2019-2022) ===")
ratios = {}
for mo, mo_name in [(9, 'Sep'), (10, 'Oct')]:
    print(f"\n{mo_name}:")
    mo_ratios = []
    for yr in range(2019, 2023):
        mo_data = sales[(sales['year'] == yr) & (sales['month'] == mo)]
        # trailing mean at start of this month
        mo_start_idx = mo_data.index[0]
        trail_val = sales.loc[mo_start_idx, 'trail60']
        mo_mean = mo_data['Revenue'].mean()
        ratio = mo_mean / trail_val if trail_val > 0 else np.nan
        mo_ratios.append(ratio)
        print(f"  {yr}: mo_mean={mo_mean:,.0f}  trail60={trail_val:,.0f}  ratio={ratio:.4f}")
    
    ratios[mo_name] = mo_ratios
    print(f"  -> min={min(mo_ratios):.4f}  p25={np.percentile(mo_ratios,25):.4f}  "
          f"median={np.median(mo_ratios):.4f}  mean={np.mean(mo_ratios):.4f}")

# Proposed: use p25 as conservative floor (will be exceeded 75% of the time)
sep_p25 = np.percentile(ratios['Sep'], 25)
oct_p25 = np.percentile(ratios['Oct'], 25)
print(f"\n=== PROPOSED DATA-DRIVEN ALPHAS ===")
print(f"  Sep: p25 = {sep_p25:.4f}")
print(f"  Oct: p25 = {oct_p25:.4f}")

# Now simulate these alphas on current submission and compare MAE vs best_624k
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
curr_orig = pd.read_csv(ROOT / 'submissions/submission.csv')
best['Date'] = pd.to_datetime(best['Date'])
curr_orig['Date'] = pd.to_datetime(curr_orig['Date'])

def mae_vs_best(df_curr, df_best):
    m = pd.merge(df_curr, df_best, on='Date', suffixes=('_c','_b'))
    return (m['Revenue_c'] - m['Revenue_b']).abs().mean()

baseline_mae = mae_vs_best(curr_orig, best)
print(f"\nBaseline MAE (current submission vs best): {baseline_mae:,.0f}")

# Test: p25 alpha for Sep, p25 for Oct
# Also test: median for both
# Also test: original 0.92 for reference
test_configs = [
    {'name': 'Original 0.92 (both)',   'sep_a': 0.92,    'oct_a': 0.92},
    {'name': 'p25 Sep + p25 Oct',       'sep_a': sep_p25, 'oct_a': oct_p25},
    {'name': 'median Sep + median Oct', 'sep_a': np.median(ratios['Sep']), 'oct_a': np.median(ratios['Oct'])},
    {'name': 'p25 Sep only (no Oct)',   'sep_a': sep_p25, 'oct_a': 999.0},  # 999 = no floor
    {'name': 'median Sep only',         'sep_a': np.median(ratios['Sep']), 'oct_a': 999.0},
]

curr_sorted = curr_orig.sort_values('Date').reset_index(drop=True)
print("\n=== SIMULATION RESULTS ===")
for cfg in test_configs:
    sim = curr_sorted['Revenue'].values.copy()
    for i in range(len(sim)):
        mo = curr_sorted['Date'].iloc[i].month
        if mo in [9, 10]:
            start = max(0, i - 60)
            if i > start:
                trailing_mean = np.mean(sim[start:i])
                alpha = cfg['sep_a'] if mo == 9 else cfg['oct_a']
                sim[i] = max(sim[i], alpha * trailing_mean)
    
    sim_df = curr_sorted[['Date']].copy()
    sim_df['Revenue'] = sim
    sim_df['COGS'] = curr_sorted['COGS']
    mae = mae_vs_best(sim_df, best)
    
    m = pd.merge(sim_df, best, on='Date', suffixes=('_c','_b'))
    sep_bias = (m[m['Date'].dt.month==9]['Revenue_c'] - m[m['Date'].dt.month==9]['Revenue_b']).mean()
    oct_bias = (m[m['Date'].dt.month==10]['Revenue_c'] - m[m['Date'].dt.month==10]['Revenue_b']).mean()
    
    print(f"\n  [{cfg['name']}]")
    print(f"    alpha_sep={cfg['sep_a']:.4f}  alpha_oct={cfg['oct_a']:.4f}")
    print(f"    MAE={mae:,.0f}  Sep_bias={sep_bias:,.0f}  Oct_bias={oct_bias:,.0f}")
