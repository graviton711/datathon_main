import pandas as pd
import numpy as np
from src.config import Config

def analyze_latent_demand():
    sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
    sales['Date'] = pd.to_datetime(sales['Date'])
    
    traffic = pd.read_parquet('data/processed/web_traffic.parquet')
    traffic['date'] = pd.to_datetime(traffic['date'])
    daily_traffic = traffic.groupby('date')['sessions'].sum().reset_index()
    
    # Calculate Intensity (Value / Monthly Mean)
    def calculate_intensity(df, date_col, val_col):
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        monthly_avg = df.groupby(['year', 'month'])[val_col].mean().reset_index(name='avg')
        df = pd.merge(df, monthly_avg, on=['year', 'month'])
        df['intensity'] = df[val_col] / (df['avg'] + 1e-6)
        return df

    sales_int = calculate_intensity(sales, 'Date', 'Revenue')
    traffic_int = calculate_intensity(daily_traffic, 'date', 'sessions')
    
    # Merge intensities
    comparison = pd.merge(
        sales_int[['Date', 'intensity']].rename(columns={'intensity': 'rev_intensity'}),
        traffic_int[['date', 'intensity']].rename(columns={'intensity': 'traffic_intensity', 'date': 'Date'}),
        on='Date'
    )
    
    # Check major holidays/peaks in 2022
    comparison['year'] = comparison['Date'].dt.year
    comparison['month'] = comparison['Date'].dt.month
    comparison['day'] = comparison['Date'].dt.day
    
    peaks = [(5,1), (9,2), (10,10), (11,11), (12,12), (30,4)]
    
    print("=== LATENT DEMAND ANALYSIS (2022 PEAKS) ===")
    print("If Traffic Intensity > Revenue Intensity, there is uncaptured potential.")
    
    results = []
    for m, d in peaks:
        val = comparison[(comparison['year'] == 2022) & (comparison['month'] == m) & (comparison['day'] == d)]
        if not val.empty:
            results.append({
                'Event': f"{d}/{m}",
                'Traffic_Int': val['traffic_intensity'].values[0],
                'Rev_Int': val['rev_intensity'].values[0],
                'Gap (T/R)': val['traffic_intensity'].values[0] / val['rev_intensity'].values[0]
            })
            
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    analyze_latent_demand()
