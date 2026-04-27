import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def check_historical_event_lift():
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Định nghĩa các ngày Black Friday lịch sử
    # 2021: 26/11, 2022: 25/11
    bf_dates = pd.to_datetime(['2021-11-26', '2022-11-25'])
    
    results = []
    for bf_date in bf_dates:
        year = bf_date.year
        # Lấy baseline là trung bình 15 ngày trước đó (tránh nhiễu đầu tháng)
        baseline = sales[(sales['Date'] >= bf_date - pd.Timedelta(days=20)) & 
                         (sales['Date'] < bf_date - pd.Timedelta(days=5))]['Revenue'].mean()
        
        actual = sales[sales['Date'] == bf_date]['Revenue'].values[0]
        lift = actual / baseline
        results.append({'Year': year, 'Lift': lift})
        
    print("--- BLACK FRIDAY HISTORICAL LIFT ---")
    print(pd.DataFrame(results))

    # Kiểm tra Double Days (11/11, 12/12)
    dd_dates = pd.to_datetime(['2021-11-11', '2021-12-12', '2022-11-11', '2022-12-12'])
    dd_results = []
    for dd_date in dd_dates:
        baseline = sales[(sales['Date'] >= dd_date - pd.Timedelta(days=10)) & 
                         (sales['Date'] < dd_date) & 
                         (sales['Date'].dt.day != 1)]['Revenue'].mean()
        actual = sales[sales['Date'] == dd_date]['Revenue'].values[0]
        dd_results.append({'Date': dd_date, 'Lift': actual/baseline})
        
    print("\n--- DOUBLE DAYS HISTORICAL LIFT ---")
    print(pd.DataFrame(dd_results))

if __name__ == "__main__":
    check_historical_event_lift()
