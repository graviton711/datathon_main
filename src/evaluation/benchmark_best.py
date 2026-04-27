import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline
from src.evaluation.evaluate import run_walk_forward_validation

def calculate_detailed_metrics(df_pred, df_best):
    """Calculates MAE, Bias, and MAPE between prediction and best reference."""
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_best['Date'] = pd.to_datetime(df_best['Date'])
    
    merged = pd.merge(df_pred, df_best, on='Date', suffixes=('_pred', '_best'))
    
    # Calculate daily errors
    merged['AE_Rev'] = np.abs(merged['Revenue_pred'] - merged['Revenue_best'])
    merged['AE_COGS'] = np.abs(merged['COGS_pred'] - merged['COGS_best'])
    merged['Total_AE'] = (merged['AE_Rev'] + merged['AE_COGS']) / 2
    
    # Calculate Bias (Pred - Best)
    merged['Bias_Rev'] = merged['Revenue_pred'] - merged['Revenue_best']
    merged['Bias_COGS'] = merged['COGS_pred'] - merged['COGS_best']
    
    # Aggregate by Quarter
    merged['Quarter'] = merged['Date'].dt.to_period('Q')
    quarterly = merged.groupby('Quarter').agg({
        'Revenue_best': 'sum',
        'Revenue_pred': 'sum',
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
    
    # Add Percentage Diff for Revenue
    quarterly['Diff_%'] = (quarterly['Revenue_pred'] / quarterly['Revenue_best'] - 1) * 100
    
    total_mae = merged['Total_AE'].mean()
    rev_mae = merged['AE_Rev'].mean()
    cogs_mae = merged['AE_COGS'].mean()
    
    # MAPE calculation (avoid division by zero)
    rev_best_nonzero = merged['Revenue_best'].replace(0, np.nan)
    mape_rev = np.nanmean(merged['AE_Rev'] / rev_best_nonzero) * 100
    
    return quarterly, total_mae, rev_mae, cogs_mae, mape_rev

def plot_comparison(df_pred, df_best, output_path):
    """Generates a comparison plot for Revenue."""
    plt.figure(figsize=(15, 7))
    plt.plot(df_best['Date'], df_best['Revenue'], label='Best Reference (624k)', color='black', alpha=0.3, linestyle='--')
    plt.plot(df_pred['Date'], df_pred['Revenue'], label='Current Pipeline', color='blue', alpha=0.7)
    
    plt.title("Revenue Comparison: Current Pipeline vs Best Reference")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_benchmark():
    print("="*70)
    print("   PIPELINE VS BEST_624K BENCHMARK (Detailed Comparison)   ")
    print("="*70)
    
    # 1. Load Reference (Best 624k)
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    if not best_path.exists():
        print(f"Error: Reference file {best_path} not found.")
        return
    
    df_best = pd.read_csv(best_path)
    df_best['Date'] = pd.to_datetime(df_best['Date'])
    
    # 2. Run Pipeline Inference
    print("\n[Step 1/2] Running Pipeline Inference (2023-2024)...")
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})
    df_pred = pipeline.predict(test_df)
    
    # 3. Calculate Proxy MAE & Detailed Metrics
    q_metrics, total_mae, rev_mae, cogs_mae, mape_rev = calculate_detailed_metrics(df_pred, df_best)
    
    print("\n--- QUARTERLY COMPARISON (vs 624k) ---")
    cols_show = ['MAE_Total', 'MAE_Rev', 'Avg_Bias', 'Diff_%']
    print(q_metrics[cols_show].to_string(formatters={'Diff_%': '{:,.2f}%'.format}))
    
    print("-" * 50)
    print(f"Overall MAE (Total)   : {total_mae:,.0f}")
    print(f"Overall MAE (Revenue) : {rev_mae:,.0f}")
    print(f"Overall MAE (COGS)    : {cogs_mae:,.0f}")
    print(f"Overall MAPE (Revenue): {mape_rev:.2f}%")
    
    # 4. Save Visualization
    plot_dir = Config.DATA_DIR / "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = plot_dir / "benchmark_vs_best.png"
    plot_comparison(df_pred, df_best, plot_path)
    print(f"\nComparison plot saved to: {plot_path}")
    
    # 5. Run Historical CV
    print("\n[Step 2/2] Running Historical Walk-Forward CV (2020-2022)...")
    run_walk_forward_validation()
    
    print("\n" + "="*70)
    print("   BENCHMARK COMPLETE   ")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()
