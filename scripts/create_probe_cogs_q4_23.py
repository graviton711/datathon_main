import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_cogs_q4_23.csv"

# Constant for probing COGS (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Q4 2023 (Oct, Nov, Dec)
    q4_23_mask = (df['Date'] >= '2023-10-01') & (df['Date'] <= '2023-12-31')
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[q4_23_mask, 'COGS'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    n_days = q4_23_mask.sum()
    
    print(f"COGS Probe for Q4 2023 created at: {PROBE_FILE}")
    print(f"Total days: {n_days}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
