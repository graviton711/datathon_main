import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_oct23.csv"

# Constant for probing (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Only October 2023
    oct_23_mask = (df['Date'] >= '2023-10-01') & (df['Date'] <= '2023-10-31')
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[oct_23_mask, 'Revenue'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    n_oct = oct_23_mask.sum()
    
    print(f"Probe submission for Oct 2023 created at: {PROBE_FILE}")
    print(f"Oct 2023 days: {n_oct}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
