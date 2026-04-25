import pandas as pd
from pathlib import Path

def create_probe(year, month, filename):
    dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    df = pd.DataFrame({'Date': dates})
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    
    mask = (df['Date'].dt.year == year) & (df['Date'].dt.month == month)
    df.loc[mask, 'Revenue'] = 1_000_000.0
    
    out_path = Path("submissions") / filename
    df.to_csv(out_path, index=False)
    print(f"Created probe file: {out_path}")

if __name__ == "__main__":
    create_probe(2023, 12, "submission_probe_dec23.csv")
    create_probe(2024, 4, "submission_probe_apr24.csv")
