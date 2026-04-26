import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline
from src.evaluation.evaluate import run_walk_forward_validation

def calculate_detailed_metrics(df_pred, df_best):
    """Calculates MAE and Bias between prediction and best reference."""
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_best['Date'] = pd.to_datetime(df_best['Date'])
    
    merged = pd.merge(df_pred, df_best, on='Date', suffixes=('_pred', '_best'))
    
    # Calculate daily AE
    merged['AE_Rev'] = np.abs(merged['Revenue_pred'] - merged['Revenue_best'])
    merged['AE_COGS'] = np.abs(merged['COGS_pred'] - merged['COGS_best'])
    merged['Total_AE'] = (merged['AE_Rev'] + merged['AE_COGS']) / 2
    
    # Calculate Bias (Pred - Best)
    merged['Bias_Rev'] = merged['Revenue_pred'] - merged['Revenue_best']
    
    # Aggregate by Quarter
    merged['Quarter'] = merged['Date'].dt.to_period('Q')
    quarterly = merged.groupby('Quarter').agg({
        'AE_Rev': 'mean',
        'AE_COGS': 'mean',
        'Total_AE': 'mean',
        'Bias_Rev': 'mean'
    }).rename(columns={
        'AE_Rev': 'MAE_Rev',
        'AE_COGS': 'MAE_COGS',
        'Total_AE': 'MAE_Total',
        'Bias_Rev': 'Avg_Bias'
    })
    
    total_mae = merged['Total_AE'].mean()
    rev_mae = merged['AE_Rev'].mean()
    cogs_mae = merged['AE_COGS'].mean()
    
    return quarterly, total_mae, rev_mae, cogs_mae

def run_benchmark():
    print("="*60)
    print("   PIPELINE VS BEST_624K BENCHMARK   ")
    print("="*60)
    
    # 1. Load Reference (Best 624k)
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    if not best_path.exists():
        print(f"Error: Reference file {best_path} not found.")
        return
    
    df_best = pd.read_csv(best_path)
    
    # 2. Run Pipeline Inference
    print("\n[Step 1/2] Running Pipeline Inference (2023-2024)...")
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})
    df_pred = pipeline.predict(test_df)
    
    # 3. Calculate Proxy MAE
    q_metrics, total_mae, rev_mae, cogs_mae = calculate_detailed_metrics(df_pred, df_best)
    
    print("\n--- TEST SET PROXY PERFORMANCE (vs 624k) ---")
    print(q_metrics.to_string())
    print("-" * 40)
    print(f"Overall MAE (Total)  : {total_mae:,.0f}")
    print(f"Overall MAE (Revenue): {rev_mae:,.0f}")
    print(f"Overall MAE (COGS)   : {cogs_mae:,.0f}")
    
    # 4. Run Historical CV
    print("\n[Step 2/2] Running Historical Walk-Forward CV (2020-2022)...")
    # This calls the existing evaluation logic
    run_walk_forward_validation()
    
    print("\n" + "="*60)
    print("   BENCHMARK COMPLETE   ")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
