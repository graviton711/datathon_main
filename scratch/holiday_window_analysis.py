import pandas as pd
import numpy as np

def analyze_holiday_signatures():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    holidays = {
        'MayDay': [(4, 30), (5, 1)],
        'ChildrenDay': [(6, 1)],
        'NationalDay': [(9, 2)]
    }
    
    results = {}
    
    for h_name, dates in holidays.items():
        daily_ratios = {i: [] for i in range(-5, 6)}
        
        for year in range(2012, 2023):
            for m, d in dates:
                h_date = pd.to_datetime(f"{year}-{m:02d}-{d:02d}", errors='coerce')
                if h_date is pd.NaT or h_date not in sales['Date'].values: continue
                
                # Baseline: average of same month excluding the window
                month_data = sales[(sales['Date'].dt.year == year) & (sales['Date'].dt.month == m)]
                window_dates = pd.date_range(h_date - pd.Timedelta(days=5), h_date + pd.Timedelta(days=5))
                baseline = month_data[~month_data['Date'].isin(window_dates)]['Revenue'].mean()
                
                if baseline == 0 or np.isnan(baseline): continue
                
                for offset in range(-5, 6):
                    target_date = h_date + pd.Timedelta(days=offset)
                    val = sales[sales['Date'] == target_date]['Revenue']
                    if not val.empty:
                        daily_ratios[offset].append(val.values[0] / baseline)
        
        # Calculate average per offset
        avg_ratios = {off: np.mean(r) for off, r in daily_ratios.items() if r}
        results[h_name] = avg_ratios

    print("=== HOLIDAY SIGNATURE ANALYSIS (Multipliers vs Monthly Baseline) ===")
    for h_name, sig in results.items():
        print(f"\n>>> {h_name}:")
        for off in range(-5, 6):
            val = sig.get(off, 1.0)
            bar = "#" * int(val * 10)
            print(f"T{off:>+2}: {val:.2f}x | {bar}")

if __name__ == "__main__":
    analyze_holiday_signatures()
