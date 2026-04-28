"""
Investigate DAMPING_Y2 calibration.
Current compound multiplier for 2024:
  Y1 (2023): base^1.0 = 1.277^1.0 = 1.277x
  Y2 (2024): 1.277 * 1.277^0.5 = 1.277 * 1.130 = 1.443x  <-- likely too high

Questions:
1. What does training data say about 2-year forward momentum?
2. What is the actual 2024 compound implied by best_624k?
3. What DAMPING_Y2 would produce the right 2024 level?
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year'] = sales['Date'].dt.year

# === 1. Historical year-over-year ANNUAL MEDIAN growth rates (post-2019) ===
ann = sales[sales['year'] >= 2019].groupby('year')['Revenue'].median()
print("=== 1. ANNUAL MEDIAN REVENUE (post-2019) ===")
print(ann.to_string())

print("\n=== 2. Y+1 GROWTH RATES ===")
yoy = {}
for i in range(1, len(ann)):
    yr, prev = ann.index[i], ann.index[i-1]
    g = ann.iloc[i] / ann.iloc[i-1]
    yoy[yr] = g
    print(f"  {prev} -> {yr}: {g:.4f}x")

print("\n=== 3. Y+2 COMPOUND vs Y+1 ALONE ===")
years = list(ann.index)
for i in range(2, len(years)):
    y0, y1, y2 = years[i-2], years[i-1], years[i]
    g1 = ann[y1] / ann[y0]  # Y+1 growth from Y0
    g2 = ann[y2] / ann[y0]  # Y+2 growth from Y0
    # If we compound g1 twice, would we get g2?
    g1_squared = g1 * g1
    damping_needed = np.log(g2/g1) / np.log(g1) if g1 != 1 else 0
    print(f"  {y0}->{y1}->{y2}: g1={g1:.4f}  g2={g2:.4f}  g1^2={g1_squared:.4f}  damping_needed={damping_needed:.4f}")

# === 4. What is the ACTUAL 2024 compound multiplier implied by best_624k? ===
print("\n=== 4. ACTUAL 2024 MULTIPLIER (best_624k) ===")
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
best['Date'] = pd.to_datetime(best['Date'])
best['year'] = best['Date'].dt.year

# Base scale from 2022 (last training year)
base_scale_2022 = float(ann[2022])
mean_2023 = best[best['year']==2023]['Revenue'].mean()
mean_2024 = best[best['year']==2024]['Revenue'].mean()

mult_2023 = mean_2023 / base_scale_2022
mult_2024 = mean_2024 / base_scale_2022
print(f"  Base scale (2022 median): {base_scale_2022:,.0f}")
print(f"  2023 mean: {mean_2023:,.0f}  -> Actual Y1 multiplier: {mult_2023:.4f}x")
print(f"  2024 mean: {mean_2024:,.0f}  -> Actual Y2 multiplier: {mult_2024:.4f}x")

# === 5. What DAMPING_Y2 gives the right multiplier? ===
print("\n=== 5. WHAT DAMPING_Y2 IS NEEDED? ===")
base_momentum = 1.277  # from pipeline output

# Current compound with various DAMPING_Y2
for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m_y1 = base_momentum ** 1.0
    m_y2 = m_y1 * (base_momentum ** d)
    print(f"  DAMPING_Y2={d:.1f}: Y1={m_y1:.4f}x  Y2={m_y2:.4f}x")

print(f"\n  Target Y1 (actual): {mult_2023:.4f}x")
print(f"  Target Y2 (actual): {mult_2024:.4f}x")

# Find optimal damping_y2
from scipy.optimize import brentq
def compound_y2(d):
    return base_momentum * (base_momentum ** d)

target_y2 = mult_2024
try:
    opt_d = brentq(lambda d: compound_y2(d) - target_y2, -2.0, 2.0)
    print(f"\n  Optimal DAMPING_Y2 to match actual Y2: {opt_d:.4f}")
except:
    print("\n  Could not find optimal DAMPING_Y2 in range [-2, 2]")

# === 6. Simulate: what if DAMPING_Y2 = optimal? ===
print("\n=== 6. SIMULATION: adjust DAMPING_Y2 on current submission ===")
curr = pd.read_csv(ROOT / 'submissions/submission.csv')
curr['Date'] = pd.to_datetime(curr['Date'])
curr['year']  = curr['Date'].dt.year

# Current multipliers
curr_m_y1 = base_momentum ** 1.0   # 2023
curr_m_y2 = curr_m_y1 * (base_momentum ** 0.5)  # 2024

# Re-scale 2024 predictions by (target_y2 / curr_m_y2)
for new_d in [0.0, -0.2, -0.4]:
    new_m_y2 = curr_m_y1 * (base_momentum ** new_d)
    scale_factor = new_m_y2 / curr_m_y2
    sim_rev = curr['Revenue'].copy()
    sim_rev[curr['year']==2024] *= scale_factor
    
    merged = pd.merge(pd.DataFrame({'Date': curr['Date'], 'Revenue': sim_rev}), 
                      best[['Date','Revenue']], on='Date', suffixes=('_c','_b'))
    mae = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
    bias_24 = merged[(merged['Date'].dt.year==2024)]['Revenue_c'].mean() - \
              merged[(merged['Date'].dt.year==2024)]['Revenue_b'].mean()
    print(f"  DAMPING_Y2={new_d:.1f}: Y2_mult={new_m_y2:.4f}x  scale={scale_factor:.4f}  MAE={mae:,.0f}  2024_bias={bias_24:,.0f}")
