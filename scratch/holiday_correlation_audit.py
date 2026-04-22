import pandas as pd
import numpy as np

def discover_holiday_correlation():
    # 1. Load Data
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 2. Define Major Holidays (Fixed dates for simplicity)
    holidays = [
        ('March_30', 3, 30),
        ('May_1', 5, 1),
        ('June_2', 6, 2),
        ('Sept_2', 9, 2),
        ('Oct_10', 10, 10),
        ('Nov_11', 11, 11),
        ('Dec_12', 12, 12)
    ]
    
    # 3. Calculate Lifts for each holiday per year
    all_lifts = []
    for yr in range(2013, 2023):
        year_lifts = {'Year': yr}
        for name, m, d in holidays:
            peak_date = pd.to_datetime(f"{yr}-{m:02d}-{d:02d}")
            if peak_date not in sales['Date'].values: continue
            
            peak_rev = sales[sales['Date'] == peak_date]['Revenue'].sum()
            baseline = sales[(sales['Date'].dt.year == yr) & (sales['Date'].dt.month == m) & (sales['Date'].dt.day != d)]['Revenue'].mean()
            year_lifts[name] = peak_rev / (baseline + 1e-6)
        all_lifts.append(year_lifts)
        
    df_lifts = pd.DataFrame(all_lifts).set_index('Year')
    
    print("=== HOLIDAY LIFT MATRIX (Yearly) ===")
    print(df_lifts.round(2))
    
    # 4. Correlation Analysis: Do holidays in the SAME year move together?
    print("\n=== CORRELATION BETWEEN HOLIDAYS WITHIN THE SAME YEAR ===")
    # We use rank correlation to see if 'High Intensity Years' are consistent
    print(df_lifts.corr(method='spearman').round(3))
    
    # 5. The "Golden Formula" Candidate:
    # If we know the intensity of the FIRST holiday (March 30), can we predict May 1st?
    if 'March_30' in df_lifts.columns and 'May_1' in df_lifts.columns:
        corr_val = df_lifts['March_30'].corr(df_lifts['May_1'])
        print(f"\nCorrelation between March 30 and May 1st Intensity: {corr_val:.3f}")

if __name__ == "__main__":
    discover_holiday_correlation()
