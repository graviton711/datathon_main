import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.features.temporal import add_time_features
from src.features.signals import add_blind_signals
from src.models.lgbm_model import LGBMModel
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("generate_submission")

def main():
    logger.info("Starting Final Submission Generation Pipeline...")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.get_merged_data()
    
    # 2. Prepare Training Data (Full set up to 2022-12-31)
    frontier_date = pd.to_datetime('2022-12-31')
    
    # Add blind signals (Static anchors frozen at end of 2022)
    df_features = add_time_features(raw_df)
    df_features = add_blind_signals(df_features, frontier_date=frontier_date)
    
    features = [
        'month', 'day', 'dayofweek', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_tet', 'is_hung_kings', 'is_holiday',
        'anchor_sessions', 'anchor_visitors', 'anchor_sentiment', 'anchor_momentum'
    ]
    
    X_train = df_features[features]
    y_train = df_features['Revenue']
    
    # 3. Calculate Time-Decay Weights (Prioritize 2021-2022)
    logger.info(f"Calculating sample weights with lambda={config.TIME_DECAY_LMBDA}")
    days_diff = (frontier_date - df_features['Date']).dt.days
    weights = np.exp(-config.TIME_DECAY_LMBDA * days_diff)
    
    # 4. Train Final Model
    logger.info("Training final model on full history...")
    model = LGBMModel(params={'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1})
    model.train(X_train, y_train, sample_weight=weights)
    
    # 5. Prepare Test Data (2023-2024)
    logger.info("Preparing test set for prediction...")
    test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': test_dates})
    
    # Apply Time Features
    test_df = add_time_features(test_df)
    
    # Apply Blind Signals (Broadcast frozen Dec 2022 state)
    # We use the anchors calculated from the training frontier
    context_sessions = df_features['anchor_sessions'].iloc[0]
    context_visitors = df_features['anchor_visitors'].iloc[0]
    context_sentiment = df_features['anchor_sentiment'].iloc[0]
    context_momentum = df_features['anchor_momentum'].iloc[0]
    
    test_df['anchor_sessions'] = context_sessions
    test_df['anchor_visitors'] = context_visitors
    test_df['anchor_sentiment'] = context_sentiment
    test_df['anchor_momentum'] = context_momentum
    
    X_test = test_df[features]
    
    # 6. Prediction
    logger.info("Generating predictions for 2023-2024...")
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)
    
    # 7. Create Submission File
    # COGS = Revenue * 0.8746 (based on 12.54% average margin)
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': np.round(preds, 2),
        'COGS': np.round(preds * (1 - 0.1254), 2)
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = config.SUBMISSIONS_DIR / f"submission_blind_v1_{timestamp}.csv"
    submission.to_csv(sub_path, index=False)
    
    logger.info(f"Submission generated successfully: {sub_path}")
    print(f"\nSUCCESS: Submission saved to {sub_path}")
    print(f"Total Prediction Rows: {len(submission)}")
    print(f"Prediction Range: {submission['Date'].min()} to {submission['Date'].max()}")
    print("-" * 50)

if __name__ == "__main__":
    main()
