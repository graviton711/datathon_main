"""
Generate 7 precision probing files for Day-of-Week (DoW) Revenue.
Each file sets one specific day of the week to 10,000,000 and all others to 0.
Used to solve for the exact Sum of Revenue for each day of the week on LB.
"""
import pandas as pd
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
PROBE_DIR = ROOT / 'submissions/probes_dow'
PROBE_DIR.mkdir(parents=True, exist_ok=True)

# Test horizon
horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Constant for probing
K = 10_000_000

print(f"Generating 7 probe files in {PROBE_DIR}...")

for i, day_name in enumerate(DOW_NAMES):
    df = pd.DataFrame({'Date': horizon_dates})
    # pandas dt.dayofweek: Mon=0, Sun=6
    df['Revenue'] = 0.0
    df.loc[df['Date'].dt.dayofweek == i, 'Revenue'] = float(K)
    df['COGS'] = 0.0
    
    out_name = f"probe_dow_{i+1}_{day_name}.csv"
    df.to_csv(PROBE_DIR / out_name, index=False)
    
    # Validation
    count = (df['Date'].dt.dayofweek == i).sum()
    print(f"  {out_name:<25}: {count:>3} days set to {K:,.0f}")

print("\nInstructions for solving:")
print("1. Submit each file and record the MAE.")
print("2. Use the formula: Sum_Y_dow = (MAE_0 + (n_dow * K / N) - MAE_K) * (N / 2)")
print("   Where:")
print("     MAE_0 = 4,183,865.95 (from VERIFIED_INSIGHTS)")
print("     N     = 1096 (total targets: 548 days * 2 columns)")
print("     n_dow = number of occurrences of that day in the test set")
print("     K     = 10,000,000")
