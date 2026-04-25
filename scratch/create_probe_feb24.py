import pandas as pd
from pathlib import Path

def create_probe_feb24():
    dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    df = pd.DataFrame({'Date': dates})
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    
    # Month 2 of 2024 (Tet year)
    mask = (df['Date'].dt.year == 2024) & (df['Date'].dt.month == 2)
    df.loc[mask, 'Revenue'] = 10_000_000.0
    
    out_path = Path("submissions/submission_probe_feb24_v2.csv")
    df.to_csv(out_path, index=False)
    print(f"Created probe file (Feb 2024 - 10M): {out_path}")

if __name__ == "__main__":
    create_probe_feb24()
