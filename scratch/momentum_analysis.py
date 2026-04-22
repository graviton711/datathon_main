import pandas as pd
import numpy as np

def analyze_growth():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # 1. Tín hiệu tăng trưởng cuối năm 2021 (Q4 2021 vs Q4 2020)
    q4_20 = sales[(sales['Date'].dt.year == 2020) & (sales['Date'].dt.month >= 10)]['Revenue'].mean()
    q4_21 = sales[(sales['Date'].dt.year == 2021) & (sales['Date'].dt.month >= 10)]['Revenue'].mean()
    growth_q4_21 = (q4_21 / q4_20) - 1
    
    print("=== LATE 2021 GROWTH SIGNAL ===")
    print(f"Q4 2020 (Avg/day): {q4_20:,.0f}")
    print(f"Q4 2021 (Avg/day): {q4_21:,.0f}")
    print(f"-> Growth Q4/2021 vs Q4/2020: {growth_q4_21*100:.2f}%\n")
    
    # 2. Compare each month of 2022 against 2021
    print("=== 2022 GROWTH MOMENTUM (vs 2021) ===")
    y21 = sales[sales['Date'].dt.year == 2021].groupby(sales['Date'].dt.month)['Revenue'].mean()
    y22 = sales[sales['Date'].dt.year == 2022].groupby(sales['Date'].dt.month)['Revenue'].mean()
    
    for month in range(1, 13):
        if month in y22.index and month in y21.index:
            rev_21 = y21[month]
            rev_22 = y22[month]
            growth = (rev_22 / rev_21) - 1
            
            # Compare with Q4 2021 signal
            diff_from_q4 = growth - growth_q4_21
            status = "SLOW DOWN" if growth < growth_q4_21 * 0.8 else ("SURGE" if growth > growth_q4_21 * 1.2 else "MAINTAIN")
            if growth < 0: status = "RECESSION"
            
            print(f"Month {month:02d}/2022: YoY Growth {growth*100:>6.2f}% | Diff from Q4/2021: {diff_from_q4*100:>6.2f}% -> {status}")
            
    # 3. 7-day rolling to pinpoint drop
    print("\n=== WEEKLY MOMENTUM (Jan to Jun 2022) ===")
    sales_21 = sales[sales['Date'].dt.year == 2021].set_index('Date').sort_index()
    sales_22 = sales[sales['Date'].dt.year == 2022].set_index('Date').sort_index()
    
    sales_21['day_of_year'] = sales_21.index.dayofyear
    sales_22['day_of_year'] = sales_22.index.dayofyear
    
    r21 = sales_21.groupby('day_of_year')['Revenue'].mean().rolling(7, min_periods=1).mean()
    r22 = sales_22.groupby('day_of_year')['Revenue'].mean().rolling(7, min_periods=1).mean()
    
    weekly_growth = []
    for week in range(1, 26):
        start_doy = (week - 1) * 7 + 1
        end_doy = week * 7
        
        mask = (r22.index >= start_doy) & (r22.index <= end_doy)
        if mask.sum() > 0:
            avg_r21 = r21[mask].mean()
            avg_r22 = r22[mask].mean()
            growth = (avg_r22 / avg_r21) - 1
            weekly_growth.append((week, growth))
            
    broken_week = -1
    for w, g in weekly_growth:
        print(f"Week {w:02d} (Day {w*7-6} to {w*7}): Growth {g*100:>6.2f}%")
        if g < growth_q4_21 * 0.5 and broken_week == -1 and w > 4:
            broken_week = w
            print("  >>> WARNING: MOMENTUM BROKE HERE!")

if __name__ == "__main__":
    analyze_growth()
