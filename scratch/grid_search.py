"""
Grid search: training cutoff year × LGBM num_leaves.
Tests 4 × 3 = 12 combinations, ~12s each = ~2.5 min total.
All runs use the same Sep+Oct floors (data-driven alpha per run).
"""
import pandas as pd
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirect verbose output
from io import StringIO
import contextlib

from pathlib import Path
ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline
import lightgbm as lgb

# Data
sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year'] = sales['Date'].dt.year

best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
best['Date'] = pd.to_datetime(best['Date'])

horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'Date': horizon_dates})

def run_experiment(cutoff_year, num_leaves, verbose=False):
    """Run one grid cell. Returns (mae_vs_best, bias_2023, bias_2024)."""
    # Patch Config
    Config.LGBM_NUM_LEAVES = num_leaves

    df_filtered = sales[sales['year'] >= cutoff_year].copy()

    pipe = ForecastingPipeline()
    # Update the model's num_leaves directly in the sklearn pipeline
    pipe.revenue_pipeline.named_steps['model'].set_params(num_leaves=num_leaves)
    pipe.cogs_pipeline.named_steps['model'].set_params(num_leaves=num_leaves)

    ctx = contextlib.redirect_stdout(StringIO()) if not verbose else contextlib.nullcontext()
    with ctx:
        pipe.fit(df_filtered)
        preds = pipe.predict(test_df.copy())

    merged = pd.merge(preds[['Date','Revenue']], best[['Date','Revenue']],
                      on='Date', suffixes=('_c','_b'))
    mae   = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
    b23   = (merged[merged['Date'].dt.year==2023]['Revenue_c'] -
             merged[merged['Date'].dt.year==2023]['Revenue_b']).mean()
    b24   = (merged[merged['Date'].dt.year==2024]['Revenue_c'] -
             merged[merged['Date'].dt.year==2024]['Revenue_b']).mean()

    return mae, b23, b24

# Grid
cutoffs    = [2016, 2018, 2019, 2020]
num_leaves = [31, 63, 127]

# Baseline (current: full data, 63 leaves)
print("Running baseline (2012+, leaves=63)...")
base_mae, base_b23, base_b24 = run_experiment(2012, 63)
print(f"Baseline: MAE={base_mae:,.0f}  Bias23={base_b23:,.0f}  Bias24={base_b24:,.0f}\n")

print(f"{'Cutoff':>8}  {'Leaves':>7}  {'MAE':>10}  {'Delta':>8}  {'Bias23':>10}  {'Bias24':>10}")
print("-" * 68)

results = []
for cut in cutoffs:
    for nl in num_leaves:
        if cut == 2012 and nl == 63:
            # Already computed as baseline
            results.append((cut, nl, base_mae, 0, base_b23, base_b24))
            print(f"{cut:>8}  {nl:>7}  {base_mae:>10,.0f}  {'(base)':>8}  {base_b23:>10,.0f}  {base_b24:>10,.0f}")
            continue
        import time
        t0 = time.time()
        mae, b23, b24 = run_experiment(cut, nl)
        elapsed = time.time() - t0
        delta = mae - base_mae
        results.append((cut, nl, mae, delta, b23, b24))
        marker = ' <--' if delta < -3000 else ''
        print(f"{cut:>8}  {nl:>7}  {mae:>10,.0f}  {delta:>+8,.0f}  {b23:>10,.0f}  {b24:>10,.0f}  ({elapsed:.0f}s){marker}")

print("\n=== TOP 5 CONFIGURATIONS ===")
results.sort(key=lambda x: x[2])
for cut, nl, mae, delta, b23, b24 in results[:5]:
    print(f"  cutoff={cut}, leaves={nl}: MAE={mae:,.0f} (delta={delta:+,.0f})")

# Restore defaults
Config.LGBM_NUM_LEAVES = 63
