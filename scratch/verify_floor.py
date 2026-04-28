"""
Verify: does a trailing-momentum floor fix the Sep/Oct 2023 underestimation?

Approach: In the recursive loop, Sep/Oct predictions are floored at
X% of the trailing 60-day mean of predictions made so far.
We simulate this post-hoc on the current submission to check the effect.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')

curr['Date'] = pd.to_datetime(curr['Date'])
best['Date'] = pd.to_datetime(best['Date'])

# --- Baseline MAE (current vs best) ---
def mae_vs_best(df_curr, df_best):
    m = pd.merge(df_curr, df_best, on='Date', suffixes=('_c','_b'))
    return (m['Revenue_c'] - m['Revenue_b']).abs().mean()

baseline_mae = mae_vs_best(curr, best)
print(f"Baseline MAE (curr vs best_624k): {baseline_mae:,.0f}")

# --- Simulate monthly floor ---
# For each day in forecast, floor = alpha * trailing_60d_mean of our own predictions
# Only apply floor in Sep and Oct (months 9, 10)

curr_sim = curr.copy().sort_values('Date').reset_index(drop=True)

# Test multiple alpha values
results = []
for alpha in [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]:
    sim = curr_sim['Revenue'].values.copy()
    
    for i in range(len(sim)):
        mo = curr_sim['Date'].iloc[i].month
        yr = curr_sim['Date'].iloc[i].year
        
        if mo in [9, 10] and yr == 2023:
            # Trailing 60-day mean of predictions made so far
            start = max(0, i - 60)
            trailing_mean = np.mean(sim[start:i]) if i > start else sim[i]
            floor_val = alpha * trailing_mean
            sim[i] = max(sim[i], floor_val)
    
    sim_df = curr_sim[['Date']].copy()
    sim_df['Revenue'] = sim
    sim_df['COGS'] = curr_sim['COGS']  # COGS unchanged
    
    mae = mae_vs_best(sim_df, best)
    improvement = baseline_mae - mae
    
    # Also check: did we fix the Sep/Oct gap?
    m = pd.merge(sim_df, best, on='Date', suffixes=('_c','_b'))
    sep23_bias = m[(m['Date'].dt.year==2023) & (m['Date'].dt.month==9)]['Revenue_c'].mean() - \
                 m[(m['Date'].dt.year==2023) & (m['Date'].dt.month==9)]['Revenue_b'].mean()
    oct23_bias = m[(m['Date'].dt.year==2023) & (m['Date'].dt.month==10)]['Revenue_c'].mean() - \
                 m[(m['Date'].dt.year==2023) & (m['Date'].dt.month==10)]['Revenue_b'].mean()
    
    results.append({
        'alpha': alpha,
        'MAE': mae,
        'delta_MAE': -improvement,
        'Sep23_bias': sep23_bias,
        'Oct23_bias': oct23_bias
    })

res_df = pd.DataFrame(results)
print("\n=== SIMULATION: Monthly Floor Effect (Sep/Oct 2023) ===")
print(res_df.to_string(index=False, float_format='{:,.0f}'.format))

# --- Also check: how many days actually hit the floor in best alpha case? ---
best_alpha = res_df.loc[res_df['MAE'].idxmin(), 'alpha']
print(f"\nBest alpha: {best_alpha}")

sim2 = curr_sim['Revenue'].values.copy()
floor_hits = 0
for i in range(len(sim2)):
    mo = curr_sim['Date'].iloc[i].month
    yr = curr_sim['Date'].iloc[i].year
    if mo in [9, 10] and yr == 2023:
        start = max(0, i - 60)
        trailing_mean = np.mean(sim2[start:i]) if i > start else sim2[i]
        floor_val = best_alpha * trailing_mean
        if sim2[i] < floor_val:
            floor_hits += 1
        sim2[i] = max(sim2[i], floor_val)

total_sep_oct = len(curr_sim[(curr_sim['Date'].dt.year==2023) & 
                              (curr_sim['Date'].dt.month.isin([9,10]))])
print(f"Days hitting floor: {floor_hits}/{total_sep_oct} ({floor_hits/total_sep_oct*100:.1f}%)")

# --- Cross-check: does the floor hurt other periods? ---
print("\n=== COLLATERAL CHECK: Bias by period after best floor ===")
sim_best_df = curr_sim[['Date']].copy()
sim_best_df['Revenue'] = sim2
sim_best_df['COGS'] = curr_sim['COGS']
m = pd.merge(sim_best_df, best, on='Date', suffixes=('_c','_b'))
m['year'] = m['Date'].dt.year
m['month'] = m['Date'].dt.month

for yr in [2023, 2024]:
    yr_m = m[m['year']==yr]
    mae  = (yr_m['Revenue_c'] - yr_m['Revenue_b']).abs().mean()
    bias = (yr_m['Revenue_c'] - yr_m['Revenue_b']).mean()
    print(f"  {yr}: MAE={mae:,.0f}  Bias={bias:,.0f}")
