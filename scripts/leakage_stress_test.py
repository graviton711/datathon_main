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

logger = setup_logger("leakage_test")

def run_stress_test():
    logger.info("Starting Leakage Stress Test...")
    
    # 1. Setup Base Data (Blind Frontier)
    loader = DataLoader()
    raw_df = loader.get_merged_data()
    frontier_date = pd.to_datetime('2021-06-30')
    
    train_raw = raw_df[raw_df['Date'] <= frontier_date].copy()
    test_raw = raw_df[raw_df['Date'] > frontier_date].copy()
    
    total_df = pd.concat([train_raw, test_raw], axis=0)
    df_features = add_time_features(total_df)
    df_features = add_blind_signals(df_features, frontier_date=frontier_date)
    
    features = [
        'month', 'day', 'dayofweek', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_tet', 'is_hung_kings', 'is_holiday',
        'anchor_sessions', 'anchor_visitors', 'anchor_sentiment', 'anchor_momentum'
    ]
    
    X_train = df_features[df_features['Date'] <= frontier_date][features]
    y_train = df_features[df_features['Date'] <= frontier_date]['Revenue']
    X_test_orig = df_features[df_features['Date'] > frontier_date][features].copy()
    y_test = df_features[df_features['Date'] > frontier_date]['Revenue']
    
    # 2. Train Model
    days_diff = (frontier_date - train_raw['Date']).dt.days
    weights = np.exp(-0.001 * days_diff)
    model = LGBMModel(params={'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1})
    model.train(X_train, y_train, sample_weight=weights)
    
    # 3. RUN SCENARIOS
    results = []
    
    # SCENARIO 0: Baseline (Control)
    preds = model.predict(X_test_orig)
    results.append({'Scenario': 'Control', 'MAE': np.mean(np.abs(y_test - preds))})
    
    # SCENARIO 1: SHUFFLE CALENDAR (Should break 'khớp')
    X_test_shuffled = X_test_orig.copy()
    # Shuffle time-based columns
    cols_to_shuffle = ['month', 'day', 'dayofweek', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'is_tet']
    for col in cols_to_shuffle:
        X_test_shuffled[col] = np.random.permutation(X_test_shuffled[col].values)
    
    preds_shuffled = model.predict(X_test_shuffled)
    results.append({'Scenario': 'Shuffle Calendar', 'MAE': np.mean(np.abs(y_test - preds_shuffled))})
    
    # SCENARIO 2: ZERO ANCHORS (Should break scale)
    X_test_no_anchors = X_test_orig.copy()
    X_test_no_anchors['anchor_sessions'] = 0
    X_test_no_anchors['anchor_visitors'] = 0
    preds_no_anchors = model.predict(X_test_no_anchors)
    results.append({'Scenario': 'Zero Anchors', 'MAE': np.mean(np.abs(y_test - preds_no_anchors))})
    
    # SCENARIO 3: SHUFFLE EVERYTHING
    X_test_chaos = X_test_orig.copy()
    for col in features:
        X_test_chaos[col] = np.random.permutation(X_test_chaos[col].values)
    preds_chaos = model.predict(X_test_chaos)
    results.append({'Scenario': 'Complete Chaos', 'MAE': np.mean(np.abs(y_test - preds_chaos))})
    
    # 4. REPORT
    res_df = pd.DataFrame(results)
    print("\n" + "!"*50)
    print("LEAKAGE STRESS TEST RESULTS")
    print("-" * 50)
    print(res_df.to_string(index=False))
    print("!"*50 + "\n")
    
    logger.info("Analysis:")
    if results[1]['MAE'] > results[0]['MAE'] * 1.5:
        logger.info("PASSED: Shuffling calendar destroyed accuracy. Model relies on seasonality logic.")
    else:
        logger.warning("FAILED: Shuffling calendar had little impact. POSSIBLE LEAKAGE DETECTED.")

if __name__ == "__main__":
    run_stress_test()
