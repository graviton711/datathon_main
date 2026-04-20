import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.features.temporal import add_time_features
from src.features.signals import add_blind_signals
from src.models.lgbm_model import LGBMModel
from src.utils.logger import setup_logger

logger = setup_logger("blind_eval")

def run_blind_test():
    logger.info("Initializing 548-Day Blind Evaluation Rig...")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.get_merged_data()
    
    # Define Intelligence Frontier (End of Training)
    # Target: 548 days before the end of data (2022-12-31)
    frontier_date = pd.to_datetime('2021-06-30')
    logger.info(f"Intelligence Frontier: {frontier_date.date()}")
    
    # 2. Split Data
    train_mask = raw_df['Date'] <= frontier_date
    test_mask = (raw_df['Date'] > frontier_date) & (raw_df['Date'] <= pd.to_datetime('2022-12-31'))
    
    train_raw = raw_df[train_mask].copy()
    test_raw = raw_df[test_mask].copy()
    
    logger.info(f"Train size: {len(train_raw)} | Test size (Horizon): {len(test_raw)}")
    
    # 3. Feature Engineering (Blind Mode)
    # Applying signals based ONLY on what was known at frontier_date
    total_df = pd.concat([train_raw, test_raw], axis=0)
    df_features = add_time_features(total_df)
    df_features = add_blind_signals(df_features, frontier_date=frontier_date)
    
    # Select deterministic & static features only
    features = [
        'month', 'day', 'dayofweek', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_tet', 'is_hung_kings', 'is_holiday',
        'anchor_sessions', 'anchor_visitors', 'anchor_sentiment', 'anchor_momentum'
    ]
    
    X_train = df_features[df_features['Date'] <= frontier_date][features]
    y_train = df_features[df_features['Date'] <= frontier_date]['Revenue']
    
    X_test = df_features[df_features['Date'] > frontier_date][features]
    y_test = df_features[df_features['Date'] > frontier_date]['Revenue']
    
    # 4. Train Model (MAE Optimized with Time-Decay Weights)
    logger.info("Training Model with Time-Decay Weights...")
    
    # Calculate Weights: exp(-lambda * days_from_frontier)
    # lambda = 0.001 -> 1 month ago is ~0.97, 1 year ago is ~0.70, 5 years ago is ~0.15
    days_diff = (frontier_date - train_raw['Date']).dt.days
    weights = np.exp(-0.001 * days_diff)
    
    model = LGBMModel(params={'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1})
    model.train(X_train, y_train, sample_weight=weights)
    
    # 5. Predict
    preds = model.predict(X_test)
    preds = np.maximum(0, preds) # Floor at 0
    
    # 6. Evaluate
    mae = np.mean(np.abs(y_test - preds))
    bias = np.mean(preds) / (np.mean(y_test) + 1e-6)
    
    # Naive Baseline for comparison (Predict mean of last year)
    naive_val = train_raw[train_raw['Date'] > (frontier_date - pd.Timedelta(days=365))]['Revenue'].mean()
    naive_mae = np.mean(np.abs(y_test - naive_val))
    
    print("\n" + "="*60)
    print(f"548-DAY BLIND TEST RESULTS (Target: 2021-07-01 to 2022-12-31)")
    print("-" * 60)
    print(f"Model MAE:      {mae:,.0f}")
    print(f"Naive MAE:      {naive_mae:,.0f}")
    print(f"Bias Ratio:     {bias:.3f}")
    print("-" * 60)
    if mae < naive_mae:
        print("RESULT: MODEL BEATS NAIVE BASELINE (Reliable)")
    else:
        print("RESULT: MODEL FAILS TO BEAT NAIVE (Overfitting Risk)")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_blind_test()
