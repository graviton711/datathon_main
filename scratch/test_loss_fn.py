"""
Quick test: MAE loss vs MSE loss (current).
Theory: MSE optimizes for E[Y] (mean), MAE optimizes for median(Y).
For right-skewed revenue distributions, mean > median → MSE overpredicts.
Since competition metric IS MAE, training with MAE loss is theoretically better.
"""
import pandas as pd
import numpy as np
import warnings
import sys
from io import StringIO
import contextlib
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline
import lightgbm as lgb

sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
sales['Date'] = pd.to_datetime(sales['Date'])
best = pd.read_csv(ROOT / 'data/best_submit/best_624k.csv')
best['Date'] = pd.to_datetime(best['Date'])
horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'Date': horizon_dates})

def run_with_objective(obj, label):
    pipe = ForecastingPipeline()
    pipe.revenue_pipeline.named_steps['model'].set_params(objective=obj)
    # COGS uses ratio (bounded 0-2), keep regression for stability
    with contextlib.redirect_stdout(StringIO()):
        pipe.fit(sales.copy())
        preds = pipe.predict(test_df.copy())
    merged = pd.merge(preds[['Date','Revenue']], best[['Date','Revenue']],
                      on='Date', suffixes=('_c','_b'))
    mae   = (merged['Revenue_c'] - merged['Revenue_b']).abs().mean()
    b23   = (merged[merged['Date'].dt.year==2023]['Revenue_c'] -
             merged[merged['Date'].dt.year==2023]['Revenue_b']).mean()
    b24   = (merged[merged['Date'].dt.year==2024]['Revenue_c'] -
             merged[merged['Date'].dt.year==2024]['Revenue_b']).mean()
    print(f"{label:<30}: MAE={mae:>10,.0f}  Bias23={b23:>+10,.0f}  Bias24={b24:>+10,.0f}")
    return mae, b23, b24

print("=== LOSS FUNCTION COMPARISON ===\n")
print(f"{'Config':<30}  {'MAE':>10}  {'Bias23':>12}  {'Bias24':>12}")
print("-" * 68)

mae_mse, b23_mse, b24_mse = run_with_objective('regression',    'regression (MSE) [current]')
mae_mae, b23_mae, b24_mae = run_with_objective('regression_l1', 'regression_l1 (MAE)')
mae_hbr, b23_hbr, b24_hbr = run_with_objective('huber',        'huber (robust)')

print(f"\n=== DELTA vs MSE baseline ===")
print(f"  MAE loss delta:   {mae_mae - mae_mse:>+10,.0f}")
print(f"  Huber loss delta: {mae_hbr - mae_mse:>+10,.0f}")
