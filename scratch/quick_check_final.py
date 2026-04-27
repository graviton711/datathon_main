import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def quick_check():
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    df_best = pd.read_csv(best_path, parse_dates=['Date'])
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})
    df_pred = pipeline.predict(test_df)
    
    merged = pd.merge(df_pred, df_best, on='Date', suffixes=('_pred', '_best'))
    merged['Residual'] = np.abs(merged['Revenue_pred'] - merged['Revenue_best'])
    merged['Quarter'] = merged['Date'].dt.to_period('Q')
    quarterly = merged.groupby('Quarter').agg({
        'Revenue_best': 'sum',
        'Revenue_pred': 'sum',
        'Residual': 'mean'
    })
    
    total_mae = np.abs(df_pred['Revenue'] - df_best['Revenue']).mean()
    print(f"\nFINAL REVENUE MAE VS BEST: {total_mae:,.0f}")
    
    # Check 2023 bias
    bias_2023 = (df_pred[df_pred['Date'].dt.year == 2023]['Revenue'].sum() / df_best[df_best['Date'].dt.year == 2023]['Revenue'].sum() - 1) * 100
    bias_2024 = (df_pred[df_pred['Date'].dt.year == 2024]['Revenue'].sum() / df_best[df_best['Date'].dt.year == 2024]['Revenue'].sum() - 1) * 100
    print(f"2023 Bias: {bias_2023:.2f}%")
    print(f"2024 Bias: {bias_2024:.2f}%")

if __name__ == "__main__":
    quick_check()
