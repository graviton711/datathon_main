import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def deep_residual_mining():
    # 1. Load Data
    best_path = Config.DATA_DIR / "best_submit" / "best_624k.csv"
    curr_path = Config.SUBMISSION_DIR / "submission.csv"
    
    if not curr_path.exists():
        print("Error: submission.csv not found.")
        return

    df_best = pd.read_csv(best_path, parse_dates=['Date'])
    df_curr = pd.read_csv(curr_path, parse_dates=['Date'])
    
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    df['Residual'] = df['Revenue_best'] - df['Revenue_curr']
    df['Rel_Err'] = df['Residual'] / (df['Revenue_curr'] + 1e-6)
    
    # 2. Analyze by Month
    print("\n--- BIAS BY MONTH ---")
    df['month'] = df['Date'].dt.month
    print(df.groupby('month')['Rel_Err'].mean().sort_values(ascending=False))

    # 3. Analyze specific known event windows in 2023
    events_2023 = {
        'Hung King King (Apr 29)': '2023-04-29',
        'Reunification/Labor (Apr 30-May 1)': '2023-04-30',
        'National Day (Sep 2)': '2023-09-02',
        'Mid-Autumn (Sep 29)': '2023-09-29',
        'Double Day 10/10': '2023-10-10',
        'Double Day 11/11': '2023-11-11',
        'Double Day 12/12': '2023-12-12',
        'Christmas (Dec 25)': '2023-12-25'
    }
    
    print("\n--- RESIDUALS AROUND KNOWN EVENTS (2023) ---")
    event_data = []
    for name, date_str in events_2023.items():
        dt = pd.to_datetime(date_str)
        # Look at window [-2, +2]
        window = df[(df['Date'] >= dt - pd.Timedelta(days=2)) & (df['Date'] <= dt + pd.Timedelta(days=2))]
        avg_rel = window['Rel_Err'].mean()
        event_data.append({'Event': name, 'Avg_Rel_Err': avg_rel})
    
    print(pd.DataFrame(event_data).sort_values('Avg_Rel_Err', ascending=False))

    # 4. Top 30 High Residual Days (Filtered)
    print("\n--- TOP 30 INDIVIDUAL DAYS (BEST > CURR) ---")
    top_days = df.sort_values('Residual', ascending=False).head(30)
    print(top_days[['Date', 'Revenue_curr', 'Revenue_best', 'Rel_Err']])

if __name__ == "__main__":
    deep_residual_mining()
