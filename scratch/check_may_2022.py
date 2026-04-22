import pandas as pd

def check_may():
    sales = pd.read_parquet('data/processed/sales.parquet')
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # May 1st across years
    may1 = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)]
    
    # May average (excluding May 1st to get a pure baseline)
    may_baseline = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day != 1)]
    may_avg = may_baseline.groupby(may_baseline['Date'].dt.year)['Revenue'].mean()
    
    print("=== MAY 1ST PERFORMANCE REPORT ===")
    for _, row in may1.iterrows():
        yr = row['Date'].year
        rev = row['Revenue']
        baseline = may_avg[yr]
        lift = rev / baseline
        print(f"Year {yr}: Revenue={rev:10,.0f} | Baseline={baseline:10,.0f} | Lift={lift:.2f}x")

if __name__ == "__main__":
    check_may()
