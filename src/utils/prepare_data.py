import pandas as pd
from pathlib import Path
import os
import sys

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import Config

def prepare_data():
    """
    Converts raw CSV files from BTC into optimized Parquet format.
    Requires 15 CSV files in data/raw/.
    """
    raw_dir = Config.RAW_DATA_DIR
    proc_dir = Config.PROCESSED_DATA_DIR
    
    os.makedirs(proc_dir, exist_ok=True)
    
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {raw_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files. Converting to Parquet...")
    
    for csv_path in csv_files:
        try:
            target_name = csv_path.stem + ".parquet"
            target_path = proc_dir / target_name
            
            # Special handling for dates if needed, otherwise generic load/save
            df = pd.read_csv(csv_path, low_memory=False)
            df.to_parquet(target_path, index=False)
            print(f"  - Converted {csv_path.name} -> {target_name}")
        except Exception as e:
            print(f"  - Failed to convert {csv_path.name}: {e}")

    print("\nData preparation complete. All files saved to data/processed/.")

if __name__ == "__main__":
    prepare_data()
