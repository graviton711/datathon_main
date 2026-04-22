import pandas as pd
import numpy as np

def analyze_momentum_transfer():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Calculate yearly aggregated metrics
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    years = sorted(sales['year'].unique())
    
    # Store Q4 signals and subsequent year actuals
    history = []
    
    for y in years:
        if y - 1 not in years:
            continue
            
        # Calculate Q4 momentum of PREVIOUS year (y-1)
        q4_prev = sales[(sales['year'] == y - 1) & (sales['month'] >= 10)]['Revenue'].sum()
        q4_prev2 = sales[(sales['year'] == y - 2) & (sales['month'] >= 10)]['Revenue'].sum() if (y - 2 in years) else np.nan
        
        q4_growth = (q4_prev / q4_prev2) - 1 if not pd.isna(q4_prev2) else np.nan
        
        # Calculate actual performance in CURRENT year (y)
        # H1 (Jan-Jun)
        h1_curr = sales[(sales['year'] == y) & (sales['month'] <= 6)]['Revenue'].sum()
        h1_prev = sales[(sales['year'] == y - 1) & (sales['month'] <= 6)]['Revenue'].sum()
        h1_growth = (h1_curr / h1_prev) - 1 if h1_prev > 0 else np.nan
        
        # H2 (Jul-Dec)
        h2_curr = sales[(sales['year'] == y) & (sales['month'] >= 7)]['Revenue'].sum()
        h2_prev = sales[(sales['year'] == y - 1) & (sales['month'] >= 7)]['Revenue'].sum()
        h2_growth = (h2_curr / h2_prev) - 1 if h2_prev > 0 else np.nan
        
        # Full Year
        fy_curr = sales[sales['year'] == y]['Revenue'].sum()
        fy_prev = sales[sales['year'] == y - 1]['Revenue'].sum()
        fy_growth = (fy_curr / fy_prev) - 1 if fy_prev > 0 else np.nan
        
        history.append({
            'Target_Year': y,
            'Signal_Q4_PrevYear': q4_growth * 100 if pd.notna(q4_growth) else np.nan,
            'Actual_H1_Growth': h1_growth * 100,
            'Actual_H2_Growth': h2_growth * 100,
            'Actual_FY_Growth': fy_growth * 100
        })
        
    df_hist = pd.DataFrame(history).dropna()
    print("=== MOMENTUM TRANSFER ANALYSIS ===")
    print("How Q4 Momentum predicts next year's growth:")
    print(df_hist.to_string(index=False, float_format="{:.2f}%".format))
    
    # Let's find the multiplier for H1 and H2
    df_hist['H1_Multiplier'] = df_hist['Actual_H1_Growth'] / df_hist['Signal_Q4_PrevYear']
    df_hist['H2_Multiplier'] = df_hist['Actual_H2_Growth'] / df_hist['Signal_Q4_PrevYear']
    
    print("\n=== MULTIPLIER PATTERNS ===")
    print("If we assume: Next_Year_Growth = Q4_Signal * Multiplier")
    print(df_hist[['Target_Year', 'H1_Multiplier', 'H2_Multiplier']].to_string(index=False, float_format="{:.2f}x".format))
    
    print("\nAverage H1 Multiplier:", df_hist['H1_Multiplier'].mean())
    print("Average H2 Multiplier:", df_hist['H2_Multiplier'].mean())
    
    # Calculate for 2023 prediction
    q4_2022 = sales[(sales['year'] == 2022) & (sales['month'] >= 10)]['Revenue'].sum()
    q4_2021 = sales[(sales['year'] == 2021) & (sales['month'] >= 10)]['Revenue'].sum()
    signal_2022 = (q4_2022 / q4_2021) - 1
    
    print("\n=== PROJECTION FOR 2023 ===")
    print(f"Q4 2022 Signal: {signal_2022*100:.2f}%")
    print(f"Predicted H1 2023 Growth: {(signal_2022 * df_hist['H1_Multiplier'].median()) * 100:.2f}% (using Median Multiplier)")
    print(f"Predicted H2 2023 Growth: {(signal_2022 * df_hist['H2_Multiplier'].median()) * 100:.2f}% (using Median Multiplier)")

if __name__ == "__main__":
    analyze_momentum_transfer()
