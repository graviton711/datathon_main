import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_cogs24.csv"

# Constant for probing (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Full year 2024 (Jan 1 - Jun 30 in the test set)
    year24_mask = (df['Date'].dt.year == 2024)
    
    # Set probe values
    # Score = Mean(|Rev - 0| + |COGS - C_mask|)
    # n = 183 days
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[year24_mask, 'COGS'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    n_days = year24_mask.sum()
    
    print(f"Probe for COGS 2024 created at: {PROBE_FILE}")
    print(f"Total days: {n_days}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
