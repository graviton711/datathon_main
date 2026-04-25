import pandas as pd
from pathlib import Path

# Paths
SAMPLE_SUB = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/raw/sample_submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
ZERO_SUB = OUTPUT_DIR / "submission_zeros.csv"

# Load sample submission
if SAMPLE_SUB.exists():
    df = pd.read_csv(SAMPLE_SUB)
    n_days = len(df)
    
    # Set all Revenue and COGS to 0
    df['Revenue'] = 0.0
    df['COGS'] = 0.0
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ZERO_SUB, index=False)
    
    print(f"Zero submission created at: {ZERO_SUB}")
    print(f"Total number of days in test set (n): {n_days}")
    print(f"Formula to calculate total actual value: Total = MAE_zeros * {n_days}")
else:
    print(f"Error: {SAMPLE_SUB} not found.")
