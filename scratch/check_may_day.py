import pandas as pd
from src.config import Config

def check_may_day():
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    # Filter for May 1st across all years
    may_days = sales[(sales['Date'].dt.month == 5) & (sales['Date'].dt.day == 1)]
    print("=== HISTORICAL REVENUE ON MAY 1ST ===")
    print(may_days[['Date', 'Revenue']].sort_values('Date'))
    
    # Check avg revenue in April and May
    print("\n=== MONTHLY AVG REVENUE (APRIL vs MAY) ===")
    monthly = sales.groupby([sales['Date'].dt.year, sales['Date'].dt.month])['Revenue'].mean()
    print(monthly.tail(24))

if __name__ == "__main__":
    check_may_day()
