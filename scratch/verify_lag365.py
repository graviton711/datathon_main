"""
Verify: do lag_365 YoY features (same-day-last-year revenue) predict better than
the current pipeline's momentum approach?

For 2023: lag_365 = actual 2022 daily revenue (from training data) — HONEST
For 2024: lag_365 = current pipeline's 2023 predictions — also HONEST (recursive)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')

# Load data
gtf = pd.read_parquet(ROOT / 'data/processed/global_tabular_features.parquet')
gtf['Date'] = pd.to_datetime(gtf['Date'])

# Aggregate to daily total (sum across categories)
daily = gtf.groupby('Date').agg(
    Revenue=('Revenue','sum'),
    lag_365=('lag_365','sum'),
    lag_730=('lag_730','sum'),
    lag_365_roll7=('lag_365_roll7','sum'),
    lag_365_roll30=('lag_365_roll30','sum'),
    traffic_lag_365=('traffic_lag_365','sum'),
    sentiment_lag_365=('sentiment_lag_365','mean'),
).reset_index()

best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
best['Date'] = pd.to_datetime(best['Date'])

curr = pd.read_csv(ROOT / 'submissions/submission.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
curr_sorted = curr.sort_values('Date').reset_index(drop=True)

# === 1. For 2023: lag_365 = same day 2022 revenue ===
# Build a map: date -> 2022 revenue (offset by 365 days)
# For forecast date D in 2023, look up D - 365 days in training data

train_daily = daily[daily['Date'] <= '2022-12-31'][['Date','Revenue']].copy()
train_daily = train_daily.set_index('Date')['Revenue']

print("=== 1. LAG_365 SIGNAL QUALITY FOR 2023 ===")

# For each 2023 forecast date, find corresponding 2022 date
forecast_dates_2023 = pd.date_range('2023-01-01', '2023-12-31', freq='D')
lag_data = []
for d in forecast_dates_2023:
    d_lag = d - pd.DateOffset(days=365)
    # Try exact match and nearby dates
    for offset in [0, 1, -1, 2, -2, 3, -3]:
        d_try = d_lag + pd.DateOffset(days=offset)
        if d_try in train_daily.index:
            lag_data.append({'Date': d, 'lag_365': train_daily[d_try]})
            break

lag_df_2023 = pd.DataFrame(lag_data)

# Merge with best_624k for 2023
best_2023 = best[best['Date'].dt.year == 2023].copy()
merged_2023 = pd.merge(lag_df_2023, best_2023[['Date','Revenue']], on='Date', suffixes=('_lag','_actual'))

# How well does lag_365 * growth_factor predict?
print(f"  Dates matched: {len(merged_2023)}")
print(f"  Lag_365 mean: {merged_2023['lag_365'].mean():,.0f}")
print(f"  Actual mean:  {merged_2023['Revenue'].mean():,.0f}")
print(f"  Required growth factor: {merged_2023['Revenue'].mean() / merged_2023['lag_365'].mean():.4f}x")

# Correlation between lag_365 and actual
corr = merged_2023['lag_365'].corr(merged_2023['Revenue'])
print(f"  Corr(lag_365, actual_2023): {corr:.4f}")

# MAE using lag_365 * growth_factor as prediction
growth_factor = merged_2023['Revenue'].mean() / merged_2023['lag_365'].mean()
merged_2023['pred_lag'] = merged_2023['lag_365'] * growth_factor
mae_lag = (merged_2023['pred_lag'] - merged_2023['Revenue']).abs().mean()
print(f"  MAE (lag_365 x {growth_factor:.3f}x vs actual): {mae_lag:,.0f}")

# Also try lag_365_roll7 for smoother signal
lag_roll_data = []
for d in forecast_dates_2023:
    d_lag = d - pd.DateOffset(days=365)
    # Get 7-day rolling avg from training data
    mask = (train_daily.index >= d_lag - pd.DateOffset(days=3)) & \
           (train_daily.index <= d_lag + pd.DateOffset(days=3))
    roll7 = train_daily[mask].mean()
    if not np.isnan(roll7):
        lag_roll_data.append({'Date': d, 'lag_roll7': roll7})

lag_roll_df = pd.DataFrame(lag_roll_data)
merged_roll = pd.merge(lag_roll_df, best_2023[['Date','Revenue']], on='Date')
growth_roll = merged_roll['Revenue'].mean() / merged_roll['lag_roll7'].mean()
merged_roll['pred_roll'] = merged_roll['lag_roll7'] * growth_roll
mae_roll = (merged_roll['pred_roll'] - merged_roll['Revenue']).abs().mean()
corr_roll = merged_roll['lag_roll7'].corr(merged_roll['Revenue'])
print(f"  MAE (lag_roll7 x {growth_roll:.3f}x): {mae_roll:,.0f}  Corr: {corr_roll:.4f}")

# Compare with current pipeline MAE for 2023
curr_2023 = curr[curr['Date'].dt.year == 2023]
best_2023_full = best[best['Date'].dt.year == 2023]
curr_best_merged = pd.merge(curr_2023, best_2023_full, on='Date', suffixes=('_c','_b'))
mae_curr_2023 = (curr_best_merged['Revenue_c'] - curr_best_merged['Revenue_b']).abs().mean()
print(f"\n  Current pipeline MAE for 2023: {mae_curr_2023:,.0f}")
print(f"  lag_365 simple model MAE:       {mae_lag:,.0f}")
print(f"  lag_roll7 model MAE:            {mae_roll:,.0f}")
improvement = mae_curr_2023 - mae_lag
print(f"  Improvement (lag_365):          {improvement:+,.0f}")

# === 2. For 2024: lag_365 = 2023 predictions (recursive) ===
print("\n=== 2. LAG_365 SIGNAL QUALITY FOR 2024 ===")
curr_2023_map = curr[curr['Date'].dt.year == 2023].set_index('Date')['Revenue']

forecast_dates_2024 = pd.date_range('2024-01-01', '2024-07-01', freq='D')
lag_data_2024 = []
for d in forecast_dates_2024:
    d_lag = d - pd.DateOffset(days=365)
    for offset in [0, 1, -1, 2, -2]:
        d_try = d_lag + pd.DateOffset(days=offset)
        if d_try in curr_2023_map.index:
            lag_data_2024.append({'Date': d, 'lag_365_pred': curr_2023_map[d_try]})
            break

lag_df_2024 = pd.DataFrame(lag_data_2024)
best_2024 = best[best['Date'].dt.year == 2024].copy()
merged_2024 = pd.merge(lag_df_2024, best_2024[['Date','Revenue']], on='Date', suffixes=('_lag','_actual'))

print(f"  Dates matched: {len(merged_2024)}")
growth_2024 = merged_2024['Revenue'].mean() / merged_2024['lag_365_pred'].mean()
print(f"  Required growth factor (2024 vs 2023 preds): {growth_2024:.4f}x")
corr_2024 = merged_2024['lag_365_pred'].corr(merged_2024['Revenue'])
print(f"  Corr(lag_365_pred, actual_2024): {corr_2024:.4f}")

merged_2024['pred_lag'] = merged_2024['lag_365_pred'] * growth_2024
mae_lag_2024 = (merged_2024['pred_lag'] - merged_2024['Revenue']).abs().mean()
curr_2024 = curr[curr['Date'].dt.year == 2024]
curr_best_24 = pd.merge(curr_2024, best_2024, on='Date', suffixes=('_c','_b'))
mae_curr_2024 = (curr_best_24['Revenue_c'] - curr_best_24['Revenue_b']).abs().mean()
print(f"  Current pipeline MAE for 2024: {mae_curr_2024:,.0f}")
print(f"  lag_365 simple model MAE:       {mae_lag_2024:,.0f}")

# === 3. Combined MAE (both years) ===
print("\n=== 3. COMBINED VERDICT ===")
# Simple weighted average
n_2023 = len(merged_2023)
n_2024 = len(merged_2024)
combined_mae_lag  = (mae_lag * n_2023 + mae_lag_2024 * n_2024) / (n_2023 + n_2024)
combined_mae_curr = (mae_curr_2023 * n_2023 + mae_curr_2024 * n_2024) / (n_2023 + n_2024)
print(f"  Current pipeline combined MAE: {combined_mae_curr:,.0f}")
print(f"  lag_365 naive model MAE:        {combined_mae_lag:,.0f}")
print(f"  delta: {combined_mae_lag - combined_mae_curr:+,.0f}")
print()
if combined_mae_lag < combined_mae_curr:
    print("  --> lag_365 alone is BETTER than current pipeline vs best_624k!")
    print("  --> Adding as feature should help significantly.")
else:
    print("  --> lag_365 alone is weaker than current pipeline.")
    print("  --> May still help as ADDITIONAL feature, but not dominant signal.")
