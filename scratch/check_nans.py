import pandas as pd
from pathlib import Path

data_path = Path('data/processed/sales.parquet')
if not data_path.exists():
    print(f"File not found: {data_path}")
else:
    df = pd.read_parquet(data_path)
    print(f"Total rows: {len(df)}")
    print(f"Total NaNs in Revenue: {df['Revenue'].isna().sum()}")
    print(f"Total NaNs in COGS: {df['COGS'].isna().sum()}")
    
    # Check specifically for the tail of 2020 which is used in evaluation
    df['Date'] = pd.to_datetime(df['Date'])
    tail_2020 = df[df['Date'] <= '2020-12-31'].tail(60)
    print(f"\nNaNs in last 60 days of 2020:")
    print(tail_2020.isna().sum())
    
    # Check for zeros in annual median which might cause division issues
    df['year'] = df['Date'].dt.year
    annual_median = df.groupby('year')['Revenue'].median()
    print(f"\nAnnual Medians:\n{annual_median}")
