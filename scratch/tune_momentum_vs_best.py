import pandas as pd
import numpy as np
import sys
from pathlib import Path
import copy

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def find_optimal_momentum_multipliers():
    # 1. Setup
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    df_best = pd.read_csv(best_path, parse_dates=['Date'])
    
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Train Base Pipeline once
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    original_momentum = copy.deepcopy(pipeline.momentum)
    print(f"Original Momentum: Base={original_momentum['base']:.4f}, Event={original_momentum['event']:.4f}")
    
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})
    
    # 3. Grid Search Multipliers
    results = []
    # Thử nghiệm các mức điều chỉnh từ 0.8x đến 1.2x
    for m_base in np.linspace(0.8, 1.2, 5):
        for m_event in np.linspace(0.8, 1.5, 8):
            test_pipeline = copy.deepcopy(pipeline)
            test_pipeline.momentum['base'] = original_momentum['base'] * m_base
            test_pipeline.momentum['event'] = original_momentum['event'] * m_event
            # Apply to categories too
            for cat in test_pipeline.momentum['categories']:
                test_pipeline.momentum['categories'][cat] = original_momentum['categories'][cat] * m_base
            
            pred = test_pipeline.predict(test_df)
            mae = np.abs(pred['Revenue'] - df_best['Revenue']).mean()
            results.append({'m_base': m_base, 'm_event': m_event, 'MAE_vs_Best': mae})
            print(f"Base Mult: {m_base:.2f}, Event Mult: {m_event:.2f} -> MAE vs Best: {mae:,.0f}")

    df_res = pd.DataFrame(results)
    best_params = df_res.loc[df_res['MAE_vs_Best'].idxmin()]
    
    print("\n--- OPTIMAL MULTIPLIERS FOUND ---")
    print(best_params)
    
    # Gợi ý cách áp dụng vào code chính
    print(f"\nRecommendation: Multiply base momentum by {best_params['m_base']:.2f} and event momentum by {best_params['m_event']:.2f}")

if __name__ == "__main__":
    find_optimal_momentum_multipliers()
