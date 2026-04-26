import pandas as pd
import numpy as np

# Paths
CURR_SUB = 'submissions/submission.csv'
BEST_SUB = 'data/best_submit/best_624k.csv'

def analyze_error_time_series():
    print("Loading submissions for analysis...")
    df_curr = pd.read_csv(CURR_SUB, parse_dates=['Date'])
    df_best = pd.read_csv(BEST_SUB, parse_dates=['Date'])
    
    # Merge
    df = pd.merge(df_curr, df_best, on='Date', suffixes=('_curr', '_best'))
    
    # Calculate Absolute Error
    df['Rev_AE'] = np.abs(df['Revenue_curr'] - df['Revenue_best'])
    df['COGS_AE'] = np.abs(df['COGS_curr'] - df['COGS_best'])
    df['Total_AE'] = (df['Rev_AE'] + df['COGS_AE']) / 2
    
    # Group by Month to see which months are the worst
    monthly_error = df.groupby(df.Date.dt.to_period('M')).agg({
        'Total_AE': 'mean',
        'Revenue_best': 'mean',
        'Revenue_curr': 'mean'
    }).reset_index()
    
    monthly_error['Error_Pct'] = (monthly_error['Total_AE'] / monthly_error['Revenue_best']) * 100
    
    print("\n--- WORST MONTHS (By MAE) ---")
    worst_months = monthly_error.sort_values('Total_AE', ascending=False).head(5)
    print(worst_months[['Date', 'Total_AE', 'Error_Pct']])
    
    # Group by Quarter
    quarterly_error = df.groupby(df.Date.dt.to_period('Q')).agg({
        'Total_AE': 'mean'
    }).reset_index()
    
    print("\n--- QUARTERLY ERROR TREND ---")
    print(quarterly_error)

    # Specific day analysis: Top 10 worst days
    print("\n--- TOP 10 WORST DAYS ---")
    worst_days = df.sort_values('Total_AE', ascending=False).head(10)
    print(worst_days[['Date', 'Revenue_best', 'Revenue_curr', 'Total_AE']])

if __name__ == '__main__':
    analyze_error_time_series()
