import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_n.csv"

# The Master Key Constant (1 Billion)
C = 1_000_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    
    # Reset all to 0
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    
    # Set only the FIRST DAY to C
    df.iloc[0, df.columns.get_loc('Revenue')] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    print(f"Master Key probe created at: {PROBE_FILE}")
    print(f"Only the first day (2023-01-01) is set to {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
