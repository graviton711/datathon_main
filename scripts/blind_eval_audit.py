import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.features.temporal import add_time_features
from src.features.signals import add_blind_signals
from src.models.lgbm_model import LGBMModel
from src.utils.logger import setup_logger

logger = setup_logger("blind_audit")

def evaluate_frontier(frontier_date, raw_df, features):
    frontier_date = pd.to_datetime(frontier_date)
    train_mask = raw_df['Date'] <= frontier_date
    # Predict 548 days ahead
    test_mask = (raw_df['Date'] > frontier_date) & (raw_df['Date'] <= frontier_date + pd.Timedelta(days=548))
    
    # Clip test mask to available data
    test_mask = test_mask & (raw_df['Date'] <= raw_df['Date'].max())
    
    train_raw = raw_df[train_mask].copy()
    test_raw = raw_df[test_mask].copy()
    
    if len(test_raw) == 0:
        return None
        
    total_df = pd.concat([train_raw, test_raw], axis=0)
    df_features = add_time_features(total_df)
    df_features = add_blind_signals(df_features, frontier_date=frontier_date)
    
    X_train = df_features[df_features['Date'] <= frontier_date][features]
    y_train = df_features[df_features['Date'] <= frontier_date]['Revenue']
    X_test = df_features[df_features['Date'] > frontier_date][features]
    y_test = df_features[df_features['Date'] > frontier_date]['Revenue']
    
    # Weights
    days_diff = (frontier_date - train_raw['Date']).dt.days
    weights = np.exp(-0.001 * days_diff)
    
    model = LGBMModel(params={'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1})
    model.train(X_train, y_train, sample_weight=weights)
    
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)
    
    # 3 Metrics calculation
    mae = np.mean(np.abs(y_test - preds))
    rmse = np.sqrt(np.mean((y_test - preds)**2))
    ss_res = np.sum((y_test - preds)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    bias = np.mean(preds) / (np.mean(y_test) + 1e-6)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'preds': preds,
        'actuals': y_test.values,
        'dates': test_raw['Date'].values
    }

def run_audit():
    loader = DataLoader()
    raw_df = loader.get_merged_data()
    
    features = [
        'month', 'day', 'dayofweek', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_tet', 'is_hung_kings', 'is_holiday',
        'anchor_sessions', 'anchor_visitors', 'anchor_sentiment', 'anchor_momentum'
    ]
    
    # 1. VISUAL TEST (Latest Horizon)
    logger.info("Running Visual Audit for 2021-07 Frontier...")
    res = evaluate_frontier('2021-06-30', raw_df, features)
    
    plt.figure(figsize=(15, 6))
    plt.plot(res['dates'], res['actuals'], label='Actual Revenue', alpha=0.5, color='gray')
    plt.plot(res['dates'], res['preds'], label='Blind Prediction', color='red', linewidth=2)
    plt.title("Blind 548-Day Forecast vs Actuals (Frontier: 2021-06-30)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('artifacts/blind_forecast_audit.png')
    logger.info("Saved forecast visualization to artifacts/blind_forecast_audit.png")
    
    # 2. MULTI-FRONTIER TEST (Robustness check)
    frontiers = ['2018-12-31', '2019-12-31', '2020-12-31', '2021-06-30']
    summary = []
    
    for f in frontiers:
        logger.info(f"Auditing Frontier: {f}...")
        r = evaluate_frontier(f, raw_df, features)
        if r:
            summary.append({
                'Frontier': f,
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2'],
                'Bias': r['bias']
            })
            
    summary_df = pd.DataFrame(summary)
    print("\n" + "="*50)
    print("ROBUSTNESS AUDIT: MULTI-FRONTIER PERFORMANCE")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    run_audit()
