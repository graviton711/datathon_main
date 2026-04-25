import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
PROBE_FILE = OUTPUT_DIR / "submission_probe_cluster_feb_sep23.csv"

# Constant for probing (10 Million)
C = 10_000_000 

if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mask: Feb 2023 to Sep 2023 (8 months)
    cluster_mask = (df['Date'] >= '2023-02-01') & (df['Date'] <= '2023-09-30')
    
    # Set probe values
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    df.loc[cluster_mask, 'Revenue'] = C
    
    # Save
    df.to_csv(PROBE_FILE, index=False)
    
    n_days = cluster_mask.sum()
    
    print(f"Cluster probe (Feb-Sep 2023) created at: {PROBE_FILE}")
    print(f"Total days in cluster: {n_days}")
    print(f"Probing constant C: {C:,}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
