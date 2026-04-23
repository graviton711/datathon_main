import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
MY_SUB = PROJECT_ROOT / "submissions" / "submission.csv"
BEST_SUB = PROJECT_ROOT / "data" / "best_submit" / "best_750k.csv"

def compare_subs():
    print("Loading submissions...")
    mine = pd.read_csv(MY_SUB)
    theirs = pd.read_csv(BEST_SUB)
    
    # Ensure same dates
    mine['Date'] = pd.to_datetime(mine['Date'])
    theirs['Date'] = pd.to_datetime(theirs['Date'])
    
    merged = pd.merge(mine, theirs, on='Date', suffixes=('_mine', '_theirs'))
    
    print(f"\n--- REVENUE COMPARISON (Sample Size: {len(merged)} days) ---")
    print(f"My Mean Revenue    : {merged['Revenue_mine'].mean():,.0f}")
    print(f"Best Mean Revenue  : {merged['Revenue_theirs'].mean():,.0f}")
    print(f"Scale Difference   : {(merged['Revenue_mine'].mean() / merged['Revenue_theirs'].mean() - 1)*100:.2f}%")
    
    mae_rev = mean_absolute_error(merged['Revenue_theirs'], merged['Revenue_mine'])
    corr_rev = merged['Revenue_mine'].corr(merged['Revenue_theirs'])
    
    print(f"MAE (vs Best)      : {mae_rev:,.0f}")
    print(f"Correlation        : {corr_rev:.4f}")
    
    print(f"\n--- COGS COMPARISON ---")
    print(f"My Mean COGS      : {merged['COGS_mine'].mean():,.0f}")
    print(f"Best Mean COGS    : {merged['COGS_theirs'].mean():,.0f}")
    print(f"Scale Difference  : {(merged['COGS_mine'].mean() / merged['COGS_theirs'].mean() - 1)*100:.2f}%")
    
    mae_cogs = mean_absolute_error(merged['COGS_theirs'], merged['COGS_mine'])
    print(f"MAE COGS (vs Best): {mae_cogs:,.0f}")
    
    print(f"\n--- TOTAL SCORE SIMULATION ---")
    print(f"Estimated Total MAE: {mae_rev + mae_cogs:,.0f}")
    
    # Check for extreme anomalies
    merged['diff_pct'] = (merged['Revenue_mine'] - merged['Revenue_theirs']).abs() / merged['Revenue_theirs']
    top_diffs = merged.sort_values('diff_pct', ascending=False).head(5)
    print("\nTop 5 Days with Largest Percentage Difference:")
    print(top_diffs[['Date', 'Revenue_mine', 'Revenue_theirs', 'diff_pct']].to_string(index=False))

if __name__ == "__main__":
    compare_subs()
