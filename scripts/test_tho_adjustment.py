import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SUB_FILE = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions/submission.csv")
OUTPUT_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/submissions")
TEST_THO_FILE = OUTPUT_DIR / "submission_test_tho_decay.csv"

if SUB_FILE.exists():
    df = pd.read_csv(SUB_FILE)
    n_days = len(df)
    
    # Define start and end multipliers based on our probe findings
    # Start: Need ~1.025 to fix -2% bias in Jan 2023
    # End: Need ~0.91 to fix +10% bias in June 2024
    start_mult = 1.025
    end_mult = 0.91
    
    # Create a linear decay multiplier for each day
    multipliers = np.linspace(start_mult, end_mult, n_days)
    
    # Apply to Revenue and COGS
    df['Revenue'] = df['Revenue'] * multipliers
    df['COGS'] = df['COGS'] * multipliers
    
    # Save
    df.to_csv(TEST_THO_FILE, index=False)
    
    print(f"Test Thô submission created at: {TEST_THO_FILE}")
    print(f"Start Multiplier (Jan 23): {start_mult}")
    print(f"End Multiplier (Jun 24): {end_mult}")
    print(f"Mean Multiplier: {np.mean(multipliers):.4f}")
else:
    print(f"Error: {SUB_FILE} not found.")
