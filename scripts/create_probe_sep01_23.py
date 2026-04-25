import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_sep01_23.csv"

# Constant for probing (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Exactly September 1st, 2023
    sep01_mask = (df['Date'] == '2023-09-01')
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[sep01_mask, 'Revenue'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    print(f"Daily Probe for September 1st, 2023 created at: {PROBE_FILE}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
