import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Using sys.executable to ensure we use the same venv
        result = subprocess.run([sys.executable, "-m"] + command.split(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        return False

def main():
    print("=== DATATHON 2026: REPRODUCE ALL RESULTS ===")
    
    # 1. Prepare Data
    if not run_command("src.utils.prepare_data", "Step 1: Preparing Data (CSV to Parquet)"):
        return

    # 2. Generate EDA Plots for Report
    if not run_command("src.utils.eda_plots", "Step 2: Generating EDA Evidence Plots"):
        return

    # 3. Evaluate (CV)
    if not run_command("src.evaluation.evaluate", "Step 3: Running Walk-Forward Cross-Validation"):
        return

    # 3. Final Pipeline (Submission)
    if not run_command("src.training.pipeline", "Step 3: Training Final Model & Generating Submission"):
        return

    print("\n" + "="*45)
    print("SUCCESS: All steps completed.")
    print("Final submission: submissions/submission.csv")
    print("="*45)

if __name__ == "__main__":
    main()
