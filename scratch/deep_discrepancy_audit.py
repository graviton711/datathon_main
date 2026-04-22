import pandas as pd
import numpy as np

def deep_audit():
    curr = pd.read_csv('submissions/submission.csv')
    best = pd.read_csv('submissions/submission_750k_bench.csv')
    
    df = curr.merge(best, on='Date', suffixes=('_curr', '_best'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['error'] = df['Revenue_best'] - df['Revenue_curr']
    df['abs_error'] = df['error'].abs()
    
    # Add time features for analysis
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dow'] = df['Date'].dt.dayofweek
    
    top_50 = df.sort_values('abs_error', ascending=False).head(50)
    
    print("=== TOP 50 DISCREPANCIES DEEP AUDIT ===")
    print(top_50[['Date', 'Revenue_curr', 'Revenue_best', 'error', 'dow']].to_string(index=False))
    
    # Analysis 1: Is it Day of Week related?
    print("\n=== ERROR BY DAY OF WEEK ===")
    print(df.groupby('dow')['error'].mean().to_string())
    
    # Analysis 2: Is it Payday related (25-31)?
    df['is_payday_zone'] = df['day'].between(25, 31)
    print("\n=== ERROR IN PAYDAY ZONE (25-31) ===")
    print(df.groupby('is_payday_zone')['error'].mean().to_string())
    
    # Analysis 3: Month specific bias
    print("\n=== SYSTEMATIC BIAS BY MONTH ===")
    print(df.groupby('month')['error'].mean().to_string())

if __name__ == "__main__":
    deep_audit()
