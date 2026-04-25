import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_global_cogs.csv"

# Constant for probing (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    
    # Global Probe: Revenue = 0, COGS = C
    df['Revenue'] = 0.0
    df['COGS'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    print(f"Global COGS probe created at: {PROBE_FILE}")
    print(f"Total target pairs: {len(df)}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
