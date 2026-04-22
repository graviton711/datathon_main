import pandas as pd
import numpy as np

def compare_signaled_days():
    curr_path = 'submissions/submission.csv'
    best_path = 'data/best_submit/best_750k.csv'
    
    df_curr = pd.read_csv(curr_path)
    df_best = pd.read_csv(best_path)
    
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['abs_diff'] = (df['Revenue_curr'] - df['Revenue_best']).abs()
    
    # Define Signaling Logic
    def get_signal_type(row):
        d, m = row['Date'].day, row['Date'].month
        
        # 1. Double Days
        if d == m: return 'Double Day'
        
        # 2. Major Holidays (Approximate)
        holidays = [(1,1), (30,4), (1,5), (2,9), (24,12), (25,12), (31,12)]
        if (d, m) in holidays: return 'Holiday'
        
        # 3. Payday Windows
        if d >= 25 or d <= 5: return 'Payday'
        
        return 'No Signal'

    df['signal_type'] = df.apply(get_signal_type, axis=1)
    
    # Filter for Signaled Days only
    signaled_df = df[df['signal_type'] != 'No Signal'].sort_values('abs_diff', ascending=False)
    
    print("=== TOP DISCREPANCIES ON SIGNALED DAYS ===")
    print(signaled_df[['Date', 'signal_type', 'Revenue_curr', 'Revenue_best', 'abs_diff']].head(20).to_string(index=False))
    
    # Summary by Signal Type
    summary = df.groupby('signal_type')['abs_diff'].mean().reset_index()
    print("\n=== MAE BY SIGNAL TYPE ===")
    print(summary)

if __name__ == "__main__":
    compare_signaled_days()
