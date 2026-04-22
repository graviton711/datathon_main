import pandas as pd
import numpy as np

def compare_submissions():
    curr_path = 'submissions/submission.csv'
    best_path = 'data/best_submit/best_750k.csv'
    
    df_curr = pd.read_csv(curr_path)
    df_best = pd.read_csv(best_path)
    
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    
    rev_mae = np.abs(df['Revenue_curr'] - df['Revenue_best']).mean()
    cogs_mae = np.abs(df['COGS_curr'] - df['COGS_best']).mean()
    total_diff = rev_mae + cogs_mae
    
    print("=== COMPARISON WITH BEST SUBMISSION (750k Bench) ===")
    print(f"Revenue Diff MAE : {rev_mae:,.0f}")
    print(f"COGS Diff MAE    : {cogs_mae:,.0f}")
    print(f"Total Diff MAE   : {total_diff:,.0f}")
    
    # Check Tet periods in 2023 and 2024
    # Tet 2023: Jan 22. Tet 2024: Feb 10.
    tet_2023 = pd.to_datetime('2023-01-22')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("\n--- Tet 2023 Period (Jan 15 - Feb 5) ---")
    mask_2023 = (df['Date'] >= '2023-01-15') & (df['Date'] <= '2023-02-05')
    print(df[mask_2023][['Date', 'Revenue_curr', 'Revenue_best']].head(10).to_string(index=False))
    
    curr_total_rev = df['Revenue_curr'].sum()
    best_total_rev = df['Revenue_best'].sum()
    print(f"\nTotal Revenue (Full Horizon):")
    print(f"Current: {curr_total_rev:,.0f}")
    print(f"Best   : {best_total_rev:,.0f}")
    print(f"Ratio  : {curr_total_rev/best_total_rev:.4f}")

    # Top 20 Discrepancies
    df['abs_diff'] = (df['Revenue_curr'] - df['Revenue_best']).abs()
    top_20 = df.sort_values('abs_diff', ascending=False).head(20)
    print("\n=== TOP 20 REVENUE DISCREPANCY DAYS ===")
    print(top_20[['Date', 'Revenue_curr', 'Revenue_best', 'abs_diff']].to_string(index=False))

if __name__ == "__main__":
    compare_submissions()
