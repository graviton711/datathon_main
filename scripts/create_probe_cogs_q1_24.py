import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_cogs_q1_24.csv"

# Constant for probing
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Q1 2024 (Jan, Feb, Mar)
    q1_24_mask = (df['Date'].dt.year == 2024) & (df['Date'].dt.month <= 3)
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[q1_24_mask, 'COGS'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    print(f"COGS Probe for Q1 2024 created at: {PROBE_FILE}")
    print(f"Probing constant C: {C:,}")
    print(f"Number of targets in block: {q1_24_mask.sum()}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
