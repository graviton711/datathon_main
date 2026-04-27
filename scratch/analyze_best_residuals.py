import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def analyze_residuals():
    # 1. Load Data
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    curr_path = Config.SUBMISSION_DIR / "submission.csv"
    
    if not curr_path.exists():
        print("Vui lòng chạy pipeline chính trước để tạo submission.csv")
        return

    df_best = pd.read_csv(best_path, parse_dates=['Date'])
    df_curr = pd.read_csv(curr_path, parse_dates=['Date'])
    
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    df['Residual'] = df['Revenue_best'] - df['Revenue_curr']
    df['Rel_Residual'] = df['Residual'] / (df['Revenue_curr'] + 1e-6)
    
    # 2. Find days with largest differences (Top 20 days where Best is higher than Curr)
    print("\n--- TOP 20 DAYS WHERE BEST IS HIGHER (Potential Missed Events) ---")
    top_missed = df.sort_values('Residual', ascending=False).head(20)
    print(top_missed[['Date', 'Revenue_curr', 'Revenue_best', 'Residual', 'Rel_Residual']])
    
    # 3. Analysis by Day of Month
    print("\n--- BIAS BY DAY OF MONTH (Potential Payday patterns) ---")
    df['dom'] = df['Date'].dt.day
    dom_bias = df.groupby('dom')['Rel_Residual'].mean()
    print(dom_bias.sort_values(ascending=False).head(10))

    # 4. Analysis by Day of Week
    print("\n--- BIAS BY DAY OF WEEK (Potential Weekly patterns) ---")
    df['dow'] = df['Date'].dt.day_name()
    dow_bias = df.groupby('dow')['Rel_Residual'].mean()
    print(dow_bias.sort_values(ascending=False))

if __name__ == "__main__":
    analyze_residuals()
