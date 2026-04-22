import pandas as pd
import numpy as np

def find_max_discrepancies():
    curr_path = 'submissions/submission.csv'
    best_path = 'data/best_submit/best_750k.csv'
    
    df_curr = pd.read_csv(curr_path)
    df_best = pd.read_csv(best_path)
    
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate absolute error for Revenue
    df['rev_error'] = np.abs(df['Revenue_curr'] - df['Revenue_best'])
    
    print("=== TOP 20 DAYS WITH LARGEST REVENUE DISCREPANCY ===")
    top_20 = df.sort_values('rev_error', ascending=False).head(20)
    print(top_20[['Date', 'Revenue_curr', 'Revenue_best', 'rev_error']].to_string(index=False))
    
    # Check for specific months
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    monthly_error = df.groupby(['year', 'month'])['rev_error'].mean().reset_index()
    
    print("\n=== AVERAGE DISCREPANCY BY MONTH ===")
    print(monthly_error.sort_values('rev_error', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    find_max_discrepancies()
