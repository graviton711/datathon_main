import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_june_23.csv"

# Constant for probing
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: June 2023
    june_23_mask = (df['Date'].dt.year == 2023) & (df['Date'].dt.month == 6)
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[june_23_mask, 'Revenue'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    print(f"Revenue Probe for June 2023 created at: {PROBE_FILE}")
    print(f"Probing constant C: {C:,}")
    print(f"Number of targets in block: {june_23_mask.sum()}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
