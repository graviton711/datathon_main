import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if not mask.any(): return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_walk_forward_validation():
    print("=== Walk-Forward 3-Fold Validation ===")
    
    # 1. Load Data
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    # 2. Define 3-Fold Walk-Forward setup
    # We use a 12-month test window for each fold to maintain consistency
    folds = [
        {'train_max_year': 2019, 'test_start': 2020, 'test_end': 2020, 'weight': 0.2},
        {'train_max_year': 2020, 'test_start': 2021, 'test_end': 2021, 'weight': 0.3},
        {'train_max_year': 2021, 'test_start': 2022, 'test_end': 2022, 'weight': 0.5},
    ]
    
    fold_metrics = []
    
    # Ensure plot directory exists
    plot_dir = Config.DATA_DIR / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for i, fold in enumerate(folds):
        fold_num = i + 1
        print(f"\n--- Fold {fold_num}: Train <= {fold['train_max_year']} | Test = {fold['test_start']} ---")
        
        train_end_date = pd.to_datetime(f"{fold['train_max_year']}-12-31")
        test_start_date = pd.to_datetime(f"{fold['test_start']}-01-01")
        test_end_date = pd.to_datetime(f"{fold['test_end']}-12-31")
        
        # Split data
        train_df = raw_sales[raw_sales['Date'] <= train_end_date].copy()
        test_df = raw_sales[(raw_sales['Date'] >= test_start_date) & (raw_sales['Date'] <= test_end_date)].copy()
        
        if len(test_df) == 0:
            print("Warning: Empty test set for this fold. Skipping.")
            continue
            
        # Initialize and Train Pipeline
        pipeline = ForecastingPipeline()
        pipeline.fit(train_df)
        
        # Predict
        print("Generating forecast for test period...")
        predictions = pipeline.predict(test_df[['Date']])
        test_df['p_Revenue'] = predictions['Revenue'].values
        test_df['p_COGS'] = predictions['COGS'].values
        
        # Metrics Calculation
        mae_rev = mean_absolute_error(test_df['Revenue'], test_df['p_Revenue'])
        mape_rev = mean_absolute_percentage_error(test_df['Revenue'], test_df['p_Revenue'])
        
        mae_cogs = mean_absolute_error(test_df['COGS'], test_df['p_COGS'])
        mape_cogs = mean_absolute_percentage_error(test_df['COGS'], test_df['p_COGS'])
        
        total_mae = mae_rev + mae_cogs
        
        metrics = {
            'Fold': fold_num,
            'Test_Period': f"{fold['test_start']}",
            'MAE_Rev': mae_rev,
            'MAPE_Rev': mape_rev,
            'MAE_COGS': mae_cogs,
            'MAPE_COGS': mape_cogs,
            'Total_MAE': total_mae,
            'Weight': fold['weight']
        }
        fold_metrics.append(metrics)
        
        # Visualization
        plt.figure(figsize=(15, 6))
        plt.plot(test_df['Date'], test_df['Revenue'], label='Actual Revenue', alpha=0.5)
        plt.plot(test_df['Date'], test_df['p_Revenue'], label='Predicted Revenue', alpha=0.8)
        plt.title(f"Fold {fold_num} Validation ({fold['test_start']}) - Revenue")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / f"val_fold_{fold_num}.png")
        plt.close()
        
        print(f"Metrics for Fold {fold_num}:")
        print(f"  Revenue MAE : {mae_rev:,.0f} (MAPE: {mape_rev:.1f}%)")
        print(f"  COGS MAE    : {mae_cogs:,.0f} (MAPE: {mape_cogs:.1f}%)")
        print(f"  TOTAL MAE   : {total_mae:,.0f}")

    # Summary
    print("\n=== Validation Summary ===")
    df_metrics = pd.DataFrame(fold_metrics)
    
    # Calculate Weighted MAE
    weighted_mae = (df_metrics['Total_MAE'] * df_metrics['Weight']).sum()
    weighted_rev_mae = (df_metrics['MAE_Rev'] * df_metrics['Weight']).sum()
    
    print(df_metrics[['Fold', 'Test_Period', 'MAE_Rev', 'MAE_COGS', 'Total_MAE', 'Weight']].to_string(index=False))
    
    print("-" * 40)
    print(f"Mean Total MAE     : {df_metrics['Total_MAE'].mean():,.0f}")
    print(f"Weighted Total MAE : {weighted_mae:,.0f} (20/30/50 split)")
    print(f"Weighted Rev MAE   : {weighted_rev_mae:,.0f}")
    print("-" * 40)
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    run_walk_forward_validation()
