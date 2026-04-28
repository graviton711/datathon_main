"""
Comprehensive ablation study: test multiple post-hoc approaches at once.
Only Rule-14 compliant approaches included.

GROUP A: Seasonal floor corrections (data-driven from training)
  A1: Oct floor at p25 alpha
  A2: Oct floor at median alpha
  A3: Nov floor at median alpha
  A4: A2 + A3 combined

GROUP B: Shape correction (lag_365 relative normalization)
  B1: lag_365_rel correction for 2023 (scale by 2022 daily shape)
  B2: lag_365_rel + blended with current (50/50)

GROUP C: Global scale adjustments from training trend
  C1: Linear trend scale-up for 2023 (+growth_delta from training)
  C2: Per-quarter 2023 scale from training YoY pattern

Each approach is applied post-hoc and MAE vs best_624k is measured.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')

# Load data
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')

curr['Date'] = pd.to_datetime(curr['Date'])
best['Date'] = pd.to_datetime(best['Date'])
sales['Date'] = pd.to_datetime(sales['Date'])

curr = curr.sort_values('Date').reset_index(drop=True)
curr['year'] = curr['Date'].dt.year
curr['month'] = curr['Date'].dt.month

sales['year'] = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
post = sales[sales['year'] >= 2019].copy()

# Baseline MAE
def mae_vs_best(sim_rev, label=''):
    merged = pd.merge(
        pd.DataFrame({'Date': curr['Date'], 'Revenue': sim_rev}),
        best[['Date','Revenue']], on='Date', suffixes=('_c','_b')
    )
    total_mae = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
    bias_23 = (merged[(merged['Date'].dt.year==2023)]['Revenue_c'] -
               merged[(merged['Date'].dt.year==2023)]['Revenue_b']).mean()
    bias_24 = (merged[(merged['Date'].dt.year==2024)]['Revenue_c'] -
               merged[(merged['Date'].dt.year==2024)]['Revenue_b']).mean()
    return total_mae, bias_23, bias_24

baseline_rev = curr['Revenue'].values.copy()
baseline_mae, b23, b24 = mae_vs_best(baseline_rev, 'Baseline')
print(f"{'Approach':<35} {'MAE':>10} {'Delta':>8} {'Bias23':>12} {'Bias24':>12}")
print(f"{'Baseline':<35} {baseline_mae:>10,.0f} {'---':>8} {b23:>12,.0f} {b24:>12,.0f}")
print("-" * 80)

results = []

# ============================================================
# GROUP A: Seasonal Floors (compute alphas from training data)
# ============================================================

def compute_floor_alpha(month_list, percentile=50):
    """Compute floor alpha for given months from training data."""
    sales_tmp = sales[sales['year'] >= 2019].copy()
    sales_tmp = sales_tmp.sort_values('Date').reset_index(drop=True)
    sales_tmp['trail60'] = sales_tmp['Revenue'].shift(1).rolling(60, min_periods=30).mean()
    ratios = []
    for mo in month_list:
        for yr in sorted(sales_tmp['year'].unique()):
            mo_rows = sales_tmp[(sales_tmp['year']==yr) & (sales_tmp['month']==mo)]
            if mo_rows.empty: continue
            trail_val = sales_tmp.loc[mo_rows.index[0], 'trail60']
            mo_mean = mo_rows['Revenue'].mean()
            if trail_val > 0 and not np.isnan(trail_val):
                ratios.append(mo_mean / trail_val)
    return float(np.percentile(ratios, percentile)) if ratios else None

def apply_floor(sim_rev, months, alpha):
    """Apply trailing momentum floor to specified months."""
    sim = sim_rev.copy()
    for i in range(len(sim)):
        if curr['month'].iloc[i] in months:
            start = max(0, i - 60)
            if i > start:
                trailing_mean = np.mean(sim[start:i])
                sim[i] = max(sim[i], alpha * trailing_mean)
    return sim

# Compute alphas from training data
oct_p25 = compute_floor_alpha([10], 25)
oct_med = compute_floor_alpha([10], 50)
nov_p25 = compute_floor_alpha([11], 25)
nov_med = compute_floor_alpha([11], 50)
feb_med = compute_floor_alpha([2], 50)
jun_p25 = compute_floor_alpha([6], 25)

print(f"  [Training-derived alphas]")
print(f"  Oct: p25={oct_p25:.4f}  median={oct_med:.4f}")
print(f"  Nov: p25={nov_p25:.4f}  median={nov_med:.4f}")
print(f"  Feb: median={feb_med:.4f}")
print(f"  Jun: p25={jun_p25:.4f}")
print()

for label, months, alpha in [
    ('A1: Oct floor p25',    [10], oct_p25),
    ('A2: Oct floor median', [10], oct_med),
    ('A3: Nov floor p25',    [11], nov_p25),
    ('A4: Nov floor median', [11], nov_med),
    ('A5: Feb floor median', [2],  feb_med),
    ('A6: Oct+Nov floor med',[10,11], oct_med),  # uses same alpha for both
    ('A7: Oct med+Nov p25',  [10,11], None),     # need custom per-month
]:
    if label == 'A7: Oct med+Nov p25':
        sim = apply_floor(baseline_rev.copy(), [10], oct_med)
        sim = apply_floor(sim, [11], nov_p25)
    else:
        sim = apply_floor(baseline_rev.copy(), months, alpha)
    mae, b23, b24 = mae_vs_best(sim)
    delta = mae - baseline_mae
    results.append((label, mae, delta, b23, b24))
    print(f"{label:<35} {mae:>10,.0f} {delta:>+8,.0f} {b23:>12,.0f} {b24:>12,.0f}")

# ============================================================
# GROUP B: lag_365 Shape Correction (relative normalization)
# ============================================================
print()
print("  [Group B: lag_365 shape correction]")

# Build lookup: date -> revenue / annual_scale (relative position within year)
ann_scale = sales.groupby('year')['Revenue'].median().to_dict()
sales_tmp = sales.copy()
sales_tmp['rel'] = sales_tmp.apply(lambda r: r['Revenue'] / ann_scale.get(r['year'], r['Revenue']), axis=1)
date_rel_map = dict(zip(sales_tmp['Date'], sales_tmp['rel']))

# Current submission: compute expected vs actual relative position for 2023
curr_2023_mean = curr[curr['year']==2023]['Revenue'].mean()
curr_2023_annual = curr_2023_mean  # use as proxy

correction_b1 = baseline_rev.copy()
correction_b2 = baseline_rev.copy()

for i in range(len(curr)):
    if curr['year'].iloc[i] == 2023:
        d = curr['Date'].iloc[i]
        d_lag = d - pd.DateOffset(days=365)
        # Find lag rel value
        lag_rel = None
        for off in [0, 1, -1, 2, -2, 3, -3]:
            d_try = d_lag + pd.DateOffset(days=off)
            if d_try in date_rel_map:
                lag_rel = date_rel_map[d_try]
                break
        if lag_rel is None:
            continue
        # Current relative position (within curr 2023)
        curr_2023_ann_median = curr[curr['year']==2023]['Revenue'].median()
        curr_rel = baseline_rev[i] / (curr_2023_ann_median + 1e-6)
        # Scale: current rel should match lag rel
        if curr_rel > 0.05:
            ratio = lag_rel / curr_rel
            # Clip ratio to avoid extreme corrections [0.8, 1.2]
            ratio = np.clip(ratio, 0.8, 1.2)
            correction_b1[i] = baseline_rev[i] * ratio
            correction_b2[i] = baseline_rev[i] * (0.5 + 0.5 * ratio)  # 50/50 blend

mae_b1, b23_b1, b24_b1 = mae_vs_best(correction_b1)
mae_b2, b23_b2, b24_b2 = mae_vs_best(correction_b2)
delta_b1 = mae_b1 - baseline_mae
delta_b2 = mae_b2 - baseline_mae
results.append(('B1: lag365_rel full',   mae_b1, delta_b1, b23_b1, b24_b1))
results.append(('B2: lag365_rel 50pct',  mae_b2, delta_b2, b23_b2, b24_b2))
print(f"{'B1: lag365_rel full':<35} {mae_b1:>10,.0f} {delta_b1:>+8,.0f} {b23_b1:>12,.0f} {b24_b1:>12,.0f}")
print(f"{'B2: lag365_rel 50pct':<35} {mae_b2:>10,.0f} {delta_b2:>+8,.0f} {b23_b2:>12,.0f} {b24_b2:>12,.0f}")

# ============================================================
# GROUP C: Training-derived trend corrections
# ============================================================
print()
print("  [Group C: Training trend corrections]")

# C1: Apply annual growth trend correction
# From training 2019-2022, fit linear trend to annual median
ann_train = sales[sales['year'] >= 2019].groupby('year')['Revenue'].median()
x = np.array(ann_train.index, dtype=float)
y = ann_train.values
slope, intercept = np.polyfit(x, y, 1)
proj_2023 = slope * 2023 + intercept
proj_2024 = slope * 2024 + intercept
actual_2022 = ann_train[2022]

# Scale 2023 predictions up by (proj_2023 / actual_2022) vs current (1.277x)
trend_scale_2023 = proj_2023 / actual_2022
trend_scale_2024 = proj_2024 / actual_2022
curr_scale_2023 = baseline_rev[curr['year']==2023].mean() / actual_2022
print(f"  Training trend: 2023 projected scale = {trend_scale_2023:.4f}x")
print(f"  Current model:  2023 actual scale     = {curr_scale_2023:.4f}x")

correction_c1 = baseline_rev.copy()
adjust_2023 = trend_scale_2023 / curr_scale_2023
correction_c1[curr['year']==2023] *= np.clip(adjust_2023, 0.9, 1.1)
mae_c1, b23_c1, b24_c1 = mae_vs_best(correction_c1)
delta_c1 = mae_c1 - baseline_mae
results.append(('C1: trend scale 2023', mae_c1, delta_c1, b23_c1, b24_c1))
print(f"{'C1: trend scale 2023':<35} {mae_c1:>10,.0f} {delta_c1:>+8,.0f} {b23_c1:>12,.0f} {b24_c1:>12,.0f}")

# C2: Q4 2023 scale-down (Q4 is traditionally weak, model over-predicts Oct-Nov)
# Historical Q4 lift (Oct-Dec) vs H1:
h1_lift = post.groupby('year').apply(lambda x: x[x['month'].isin([1,2,3,4,5,6])]['Revenue'].mean() /
                                               x['Revenue'].mean()).mean()
q4_lift = post.groupby('year').apply(lambda x: x[x['month'].isin([10,11,12])]['Revenue'].mean() /
                                               x['Revenue'].mean()).mean()
print(f"  Historical H1 lift: {h1_lift:.4f}x  Q4 lift: {q4_lift:.4f}x")

correction_c2 = baseline_rev.copy()
curr_q4_mean = baseline_rev[(curr['year']==2023) & (curr['month'].isin([10,11,12]))].mean()
expected_q4 = baseline_rev[curr['year']==2023].mean() * q4_lift
if curr_q4_mean > 0:
    q4_adj = np.clip(expected_q4 / curr_q4_mean, 0.85, 1.15)
    mask = (curr['year']==2023) & (curr['month'].isin([10,11,12]))
    correction_c2[mask] *= q4_adj
mae_c2, b23_c2, b24_c2 = mae_vs_best(correction_c2)
delta_c2 = mae_c2 - baseline_mae
results.append(('C2: Q4 2023 scale',    mae_c2, delta_c2, b23_c2, b24_c2))
print(f"{'C2: Q4 2023 scale':<35} {mae_c2:>10,.0f} {delta_c2:>+8,.0f} {b23_c2:>12,.0f} {b24_c2:>12,.0f}")

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 80)
print("SUMMARY: Approaches that IMPROVE vs baseline")
print("=" * 80)
improved = [(l, mae, d, b23, b24) for l, mae, d, b23, b24 in results if d < 0]
improved.sort(key=lambda x: x[2])
if improved:
    for label, mae, delta, b23, b24 in improved:
        print(f"  {label:<35} MAE={mae:>10,.0f}  delta={delta:>+8,.0f}  Bias23={b23:>10,.0f}  Bias24={b24:>10,.0f}")
else:
    print("  None of the approaches improved over baseline.")
    print("  The pipeline is likely at its honest ceiling.")
