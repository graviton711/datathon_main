import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.config import config
from src.data.loader import DataLoader
from src.features.temporal import add_time_features
from src.features.signals import add_signal_features
from src.models.lgbm_model import LGBMModel
from src.evaluation.backtest import TimeSeriesBacktester

def main():
    logger = setup_logger()
    logger.info("Initializing Datathon 2026 Evaluation Pipeline...")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.get_merged_data()
    
    # 2. Feature Engineering
    df = add_time_features(raw_df)
    df = add_signal_features(df)
    
    # 3. Define Analysis Scope
    # IMPORTANT: We exclude 'year' to measure 'True Baseline' performance
    features = [
        'month', 'day', 'dayofweek', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'sessions_lag1', 'visitors_lag1', 'traffic_momentum',
        'sentiment_roll30', 'sentiment_change_30d'
    ]
    
    logger.info(f"Using {len(features)} features: {features}")
    
    # 4. Run Backtest
    backtester = TimeSeriesBacktester()
    results = backtester.run_eval(
        model_class=LGBMModel,
        df=df,
        features=features
    )
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
