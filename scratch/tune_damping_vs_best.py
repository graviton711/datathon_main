import pandas as pd
import numpy as np
import sys
from pathlib import Path
import copy

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.training.pipeline import ForecastingPipeline

def tune_damping_vs_best():
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    df_best = pd.read_csv(best_path, parse_dates=['Date'])
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    pipeline = ForecastingPipeline()
    pipeline.fit(sales)
    
    horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    test_df = pd.DataFrame({'Date': horizon_dates})

    results = []
    # Grid Search cho Damping Y1 (2023) và Damping Y2 (2024)
    for d1 in [0.9, 1.0, 1.1]: # Thử nghiệm cả việc tăng trưởng mạnh hơn (1.1)
        for d2 in [0.3, 0.4, 0.5, 0.6]:
            # Tạm thời ghi đè Config bằng cách monkeypatch hoặc sửa logic predict
            # Ở đây tôi sẽ sửa trực tiếp logic tính toán multipliers trong một hàm local
            
            def compute_custom_multipliers(pipeline_obj, d1_val, d2_val):
                max_train_year = pipeline_obj.momentum['max_train_year']
                year_multipliers = {}
                cat_multipliers = {cat: {} for cat in pipeline_obj.momentum['categories']}
                running_m_base, running_m_event = 1.0, 1.0
                cat_running_m = {cat: 1.0 for cat in pipeline_obj.momentum['categories']}
                
                for yr in [2023, 2024]:
                    years_out = yr - max_train_year
                    damp_global = d1_val if years_out <= 1 else d2_val
                    
                    running_m_base *= (pipeline_obj.momentum['base'] ** damp_global)
                    running_m_event *= (pipeline_obj.momentum['event'] ** damp_global)
                    year_multipliers[yr] = {'base': running_m_base, 'event': running_m_event}
                    
                    for cat, mom in pipeline_obj.momentum['categories'].items():
                        cat_running_m[cat] *= (mom ** damp_global)
                        cat_multipliers[cat][yr] = cat_running_m[cat]
                return year_multipliers, cat_multipliers

            # Ghi đè hàm _compute_forecast_multipliers của pipeline
            original_method = pipeline._compute_forecast_multipliers
            pipeline._compute_forecast_multipliers = lambda h_yrs, m_yr: compute_custom_multipliers(pipeline, d1, d2)
            
            pred = pipeline.predict(test_df)
            mae = np.abs(pred['Revenue'] - df_best['Revenue']).mean()
            results.append({'Damping_Y1': d1, 'Damping_Y2': d2, 'MAE': mae})
            print(f"Damping Y1: {d1}, Y2: {d2} -> MAE vs Best: {mae:,.0f}")
            
            # Restore
            pipeline._compute_forecast_multipliers = original_method

    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['MAE'].idxmin()]
    print("\n--- BEST DAMPING PARAMS ---")
    print(best)

if __name__ == "__main__":
    tune_damping_vs_best()
