import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def convert_all():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(RAW_DIR.glob("*.csv"))
    
    print(f"Starting conversion of {len(csv_files)} files to Parquet (PyArrow)...")
    start_total = time.time()
    
    for csv_file in tqdm(csv_files):
        try:
            # Use pyarrow for robust type inference
            df = pd.read_csv(csv_file, engine='pyarrow')
            parquet_file = PROCESSED_DIR / (csv_file.stem + ".parquet")
            
            # Save using pyarrow engine
            df.to_parquet(parquet_file, engine='pyarrow', index=False)
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
            
    end_total = time.time()
    print(f"\nConversion complete! Total time: {end_total - start_total:.2f}s")
    print(f"Files saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    convert_all()
