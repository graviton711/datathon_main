
import pandas as pd
import numpy as np
from src.training.pipeline import ForecastingPipeline
from src.config import Config

def debug_january_2024():
    # 1. Load data
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Train pipeline
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    # 3. Predict Jan 2024
    jan_dates = pd.date_range(start='2024-01-01', end='2024-02-15', freq='D')
    test_df = pd.DataFrame({'Date': jan_dates})
    
    # We need to capture the intermediate steps in predict to see what's happening
    # Let's manually run a bit of predict logic
    event_map = pipeline.revenue_pipeline.named_steps['features'].event_score_map
    
    print(f"Momentum: {pipeline.momentum}")
    print(f"Base Scale Rev: {pipeline.base_scale_rev}")
    
    results = []
    
    # Simulate the loop in predict
    history = pipeline.history_norm.copy()
    max_train_year = pipeline.momentum['max_train_year']
    base_m = pipeline.momentum['base']
    event_m = pipeline.momentum['event']
    
    for _, row in test_df.iterrows():
        curr_date = row['Date']
        recent_hist = history.tail(Config.REC_LAG_WINDOW).copy()
        dummy = pd.DataFrame({'Date': [curr_date], 'Revenue': [0], 'COGS': [0]})
        temp_df = pd.concat([recent_hist, dummy], ignore_index=True)
        
        # Add lags
        temp_df = pipeline._add_lags(temp_df)
        current_features = temp_df.iloc[[-1]][pipeline.feature_cols].copy()
        
        # Predict norm
        raw_norm_rev = pipeline.revenue_pipeline.predict(current_features)[0]
        
        # Get features for debugging (need to transform separately to see them)
        features_extractor = pipeline.revenue_pipeline.named_steps['features']
        X_feat_debug = features_extractor.transform(current_features)
        
        # Scaling
        years_out = curr_date.year - max_train_year
        is_event = (curr_date.day >= 25) | (curr_date.dayofweek == 2) | \
                   ((curr_date.month, curr_date.day) in event_map)
        
        damping_factor = 0.5 ** (years_out - 1)
        effective_m = (event_m if is_event else base_m) ** damping_factor
        
        final_rev = raw_norm_rev * pipeline.base_scale_rev * effective_m
        
        results.append({
            'Date': curr_date,
            'days_to_tet': X_feat_debug['days_to_tet'].values[0],
            'event_score': X_feat_debug['event_score'].values[0],
            'is_event': is_event,
            'effective_m': effective_m,
            'raw_norm_rev': raw_norm_rev,
            'final_rev': final_rev
        })
        
        # Update history
        new_row = pd.DataFrame({'Date': [curr_date], 'Revenue': [raw_norm_rev], 'COGS': [0]})
        history = pd.concat([history, new_row], ignore_index=True)

    res_df = pd.DataFrame(results)
    print("\nFull predictions for Jan 2024 and Early Feb 2024:")
    pd.set_option('display.max_rows', None)
    print(res_df.to_string())

if __name__ == "__main__":
    debug_january_2024()
