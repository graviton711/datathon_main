import pandas as pd
import numpy as np
from src.config import Config

class MasterLoader:
    """
    Class to handle loading of core sales data.
    Streamlined: Removed unused web_traffic and placeholder attributes.
    """
    def __init__(self):
        self.sales = None

    def load_sales_data(self):
        """
        Loads the essential processed sales parquet file.
        """
        print(f"Loading processed sales data from {Config.SALES_TRAIN_FILE}...")
        self.sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
        
        # Ensure Date column is datetime
        self.sales[Config.DATE_COL] = pd.to_datetime(self.sales[Config.DATE_COL])
        
        return self.sales

    def get_master_dataframe(self):
        """
        Returns the sales data. Maintained method name for compatibility.
        """
        if self.sales is None:
            self.load_sales_data()
            
        return self.sales

if __name__ == "__main__":
    loader = MasterLoader()
    df = loader.get_master_dataframe()
    print(f"Master DataFrame columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(df.head())
