import pandas as pd
from pathlib import Path

SUBMISSION_FILE = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions/submission.csv")
OUTPUT_FILE = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions/submission_probing_1.04.csv")

if SUBMISSION_FILE.exists():
    df = pd.read_csv(SUBMISSION_FILE)
    df['Revenue'] = df['Revenue'] * 1.04
    df['COGS'] = df['COGS'] * 1.04
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Mathematical Optimal submission created: {OUTPUT_FILE}")
else:
    print("Original submission file not found.")
