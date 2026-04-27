import pandas as pd
import numpy as np
from src.config import Config

class DataWeighting:
    """
    Handles sample weight calculation based on recency and anomaly detection.
    Extracted from ForecastingPipeline for better modularity.
    """
    
    @staticmethod
    def apply_weights(df: pd.DataFrame, annual_scales_rev: dict):
        """
        Calculates weights based on:
        1. Recency (Trust recent data more)
        2. Efficiency Consistency (Discount anomalies in Rev/Sessions ratio)
        """
        print("Calculating Data-driven Sample Weights...")
        df = df.copy()
        
        # A. Recency Weight
        days_from_start = (df['Date'] - df['Date'].min()).dt.days
        recency_weight = np.exp((days_from_start - days_from_start.max()) / (365.0 * Config.DECAY_HALF_LIFE_YEARS))
        
        # B. Anomaly Weight (Efficiency Ratio context)
        try:
            traffic = pd.read_parquet(Config.WEB_TRAFFIC_FILE)
            traffic['date'] = pd.to_datetime(traffic['date'])
            daily_traffic = traffic.groupby('date')['sessions'].sum()
            
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['sessions'] = df['Date'].map(daily_traffic).fillna(1e-6)
            df['raw_rev'] = df['Revenue'] * df['year'].map(annual_scales_rev)
            
            monthly_stats = df.groupby(['year', 'month']).agg({'raw_rev': 'sum', 'sessions': 'sum'}).reset_index()
            monthly_stats['eff_ratio'] = monthly_stats['raw_rev'] / (monthly_stats['sessions'] + 1e-6)
            
            yearly_meta = monthly_stats.groupby('year')['eff_ratio'].agg(['median', 'std']).reset_index()
            yearly_meta.columns = ['year', 'y_median', 'y_std']
            
            monthly_stats = pd.merge(monthly_stats, yearly_meta, on='year')
            monthly_stats['z_score'] = (monthly_stats['eff_ratio'] - monthly_stats['y_median']).abs() / (monthly_stats['y_std'] + 1e-6)
            monthly_stats['outlier_weight'] = 1.0 / (1.0 + monthly_stats['z_score'])
            
            weight_map = monthly_stats.set_index(['year', 'month'])['outlier_weight'].to_dict()
            outlier_weights = pd.MultiIndex.from_arrays([df['year'], df['month']]).map(weight_map).fillna(1.0)
        except Exception as e:
            print(f"Warning: Anomaly weighting failed ({e}). Using 1.0.")
            outlier_weights = 1.0
            
        df['sample_weight'] = recency_weight * outlier_weights
        
        # Clean up temp columns
        cols_to_drop = ['raw_rev', 'sessions', 'year', 'month']
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns])
