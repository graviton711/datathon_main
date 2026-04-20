import pandas as pd
from src.config import config

class DataLoader:
    def __init__(self):
        self.raw_dir = config.RAW_DATA_DIR

    def load_sales(self):
        """Loads the analytical sales data."""
        df = pd.read_csv(self.raw_dir / "sales.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def load_table(self, table_name):
        """Loads a specific table from raw data."""
        return pd.read_csv(self.raw_dir / f"{table_name}.csv")
