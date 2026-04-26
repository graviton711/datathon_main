import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
PIPELINE_FILE = 'src/training/pipeline.py'
SUBMISSION_FILE = 'submissions/submission.csv'
BEST_SUB = 'data/best_submit/best_624k.csv'
LOG_FILE = 'scratch/optimization_log.json'

def get_current_mae():
    if not os.path.exists(SUBMISSION_FILE):
        return float('inf')
    df_curr = pd.read_csv(SUBMISSION_FILE)
    df_best = pd.read_csv(BEST_SUB)
    mae = np.mean(np.abs(df_curr['Revenue'] - df_best['Revenue']))
    return mae

def run_pipeline():
    result = subprocess.run(['python', '-m', 'src.training.pipeline'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Pipeline failed: {result.stderr}")
        return False
    return True

def update_param(param_name, value):
    with open(PIPELINE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple replacement logic for known parameters
    if param_name == 'w_aov':
        # Find 'w_aov': 0.0 or similar in __init__
        import re
        content = re.sub(r"'w_aov': -?\d+\.\d+", f"'w_aov': {value}", content)
    
    with open(PIPELINE_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

def optimize():
    best_mae = get_current_mae()
    print(f"Initial MAE: {best_mae:.2f}")
    
    trials = [
        ('w_aov', 0.5),
        ('w_aov', 1.0),
        ('w_aov', 1.5),
        ('w_aov', 2.0),
    ]
    
    results = []
    
    for param, val in trials:
        print(f"\nTrying {param} = {val}...")
        update_param(param, val)
        if run_pipeline():
            current_mae = get_current_mae()
            print(f"Resulting MAE: {current_mae:.2f}")
            results.append({'param': param, 'val': val, 'mae': current_mae})
            
            if current_mae < best_mae:
                best_mae = current_mae
                print(f"*** NEW BEST MAE: {best_mae:.2f} ***")
            else:
                print("No improvement.")
        else:
            print("Trial failed.")

    # Save results
    with open(LOG_FILE, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    optimize()
