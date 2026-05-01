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
    mask = y_true > 1e-6
    if not mask.any(): return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_walk_forward_validation():
    print("=== Walk-Forward 3-Fold Validation ===")
    
    # 1. Load Data
    raw_sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    raw_sales['Date'] = pd.to_datetime(raw_sales['Date'])
    
    # 2. Define 3-Fold Walk-Forward setup
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
            
        # 3. Naive Baseline (Last Year aligned by Day of Week: 52 weeks = 364 days)
        test_df['ly_Revenue'] = test_df['Date'].map(raw_sales.set_index(raw_sales['Date'] + pd.Timedelta(days=364))['Revenue']).ffill()
        test_df['ly_COGS'] = test_df['Date'].map(raw_sales.set_index(raw_sales['Date'] + pd.Timedelta(days=364))['COGS']).ffill()
        
        # Initialize and Train Pipeline
        pipeline = ForecastingPipeline()
        pipeline.fit(train_df)
        
        # Predict
        print("Generating forecast for test period...")
        predictions = pipeline.predict(test_df[['Date']])
        test_df['p_Revenue'] = predictions['Revenue'].values
        test_df['p_COGS'] = predictions['COGS'].values
        
        # Metrics Calculation (Model)
        mae_rev = mean_absolute_error(test_df['Revenue'], test_df['p_Revenue'])
        mae_cogs = mean_absolute_error(test_df['COGS'], test_df['p_COGS'])
        
        # Business Metric: Gross Profit MAE
        test_df['Profit'] = test_df['Revenue'] - test_df['COGS']
        test_df['p_Profit'] = test_df['p_Revenue'] - test_df['p_COGS']
        mae_profit = mean_absolute_error(test_df['Profit'], test_df['p_Profit'])
        
        # Baseline Metrics
        mae_rev_ly = mean_absolute_error(test_df['Revenue'], test_df['ly_Revenue'].fillna(0))
        
        lift_over_ly = (1 - mae_rev / mae_rev_ly) * 100 if mae_rev_ly > 0 else 0
        
        metrics = {
            'Fold': fold_num,
            'Test_Period': f"{fold['test_start']}",
            'MAE_Rev': mae_rev,
            'MAE_COGS': mae_cogs,
            'MAE_Profit': mae_profit,
            'MAE_LY_Rev': mae_rev_ly,
            'Lift_LY_%': lift_over_ly,
            'Weight': fold['weight']
        }
        fold_metrics.append(metrics)
        
        # Visualization (Optimized for Print/Report)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 5))
        plt.plot(test_df['Date'], test_df['Revenue'], label='Actual Revenue', alpha=0.5, color='gray', linewidth=1)
        plt.plot(test_df['Date'], test_df['p_Revenue'], label='Predicted Revenue', alpha=0.9, color='#1f77b4', linewidth=1.5)
        plt.plot(test_df['Date'], test_df['ly_Revenue'], label='Naive Baseline', linestyle=':', alpha=0.6, color='red')
        
        plt.title(f"Fold {fold_num} Validation ({fold['test_start']})", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Revenue (VND)", fontsize=12)
        plt.legend(fontsize=10, loc='upper right', frameon=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        # Save to both data/plots and reports/ for LaTeX
        plt.savefig(plot_dir / f"val_fold_{fold_num}.png", dpi=300, bbox_inches='tight')
        plt.savefig(PROJECT_ROOT / "reports" / f"val_fold_{fold_num}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics for Fold {fold_num}:")
        print(f"  Revenue MAE   : {mae_rev:,.0f} (LY Baseline: {mae_rev_ly:,.0f} | Lift: {lift_over_ly:.1f}%)")
        print(f"  Profit MAE    : {mae_profit:,.0f}")
        print(f"  COGS MAE      : {mae_cogs:,.0f}")

    # Summary
    print("\n=== Validation Summary ===")
    df_metrics = pd.DataFrame(fold_metrics)
    
    # Calculate Weighted Metrics
    w_rev_mae = (df_metrics['MAE_Rev'] * df_metrics['Weight']).sum()
    w_profit_mae = (df_metrics['MAE_Profit'] * df_metrics['Weight']).sum()
    w_lift = (df_metrics['Lift_LY_%'] * df_metrics['Weight']).sum()
    
    print(df_metrics[['Fold', 'Test_Period', 'MAE_Rev', 'MAE_Profit', 'Lift_LY_%', 'Weight']].to_string(index=False))
    
    print("-" * 50)
    print(f"Weighted Rev MAE    : {w_rev_mae:,.0f}")
    print(f"Weighted Profit MAE : {w_profit_mae:,.0f}")
    print(f"Weighted Lift vs LY : {w_lift:.2f}%")
    print("-" * 50)
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    run_walk_forward_validation()

