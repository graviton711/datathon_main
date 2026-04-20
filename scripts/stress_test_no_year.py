from src.data.loader import loader
from src.features.temporal import add_time_features
from src.models.lgbm_model import LGBMModel
from src.evaluation.backtest import TimeSeriesBacktester
from src.utils.logger import setup_logger

logger = setup_logger("stress_test")

def main():
    logger.info("Running Stress Test: NO 'year' feature...")
    
    # 1. Load merged data
    df = loader.get_merged_data()
    
    # 2. Add basic temporal features
    df = add_time_features(df, date_col='Date')
    
    # 3. Define features (REMOVING 'year')
    features = ['month', 'day', 'dayofweek', 'is_weekend']
    
    # 4. Setup backtester
    backtester = TimeSeriesBacktester(date_col='Date', target_col='Revenue')
    
    # 5. Run evaluation using basic LGBM
    results = backtester.run_eval(
        model_class=LGBMModel,
        df=df,
        features=features
    )
    
    logger.info("Stress Test Complete.")

if __name__ == "__main__":
    main()
