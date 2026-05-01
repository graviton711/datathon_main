import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import Config

def generate_eda_report_plots():
    """Generates the specific EDA plots required for the final technical report."""
    raw_dir = Config.RAW_DATA_DIR
    report_dir = Config.PROJECT_ROOT / "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    print("Generating EDA plots for report...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Traffic Conversion Catastrophe (12_traffic_conversion.png)
    try:
        traffic = pd.read_csv(raw_dir / "web_traffic.csv")
        orders = pd.read_csv(raw_dir / "orders.csv")
        
        traffic['date'] = pd.to_datetime(traffic['date'])
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        
        daily_traffic = traffic.groupby('date')['sessions'].sum().reset_index()
        daily_orders = orders.groupby('order_date').size().reset_index(name='order_count')
        
        merged = pd.merge(daily_traffic, daily_orders, left_on='date', right_on='order_date', how='left').fillna(0)
        merged['CR'] = (merged['order_count'] / merged['sessions']) * 100
        merged['Year'] = merged['date'].dt.year
        
        # Monthly Resample for smoother plot
        monthly = merged.set_index('date').resample('ME').agg({'sessions': 'sum', 'CR': 'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        
        sns.lineplot(data=monthly, x='date', y='sessions', ax=ax1, color='gray', alpha=0.5, label='Sessions')
        sns.lineplot(data=monthly, x='date', y='CR', ax=ax2, color='red', label='Conversion Rate (%)')
        
        ax1.set_ylabel('Total Sessions')
        ax2.set_ylabel('Conversion Rate (%)')
        plt.title('The Conversion Catastrophe (2018-2019)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(report_dir / "12_traffic_conversion.png", dpi=300)
        plt.close()
        print("  - Saved 12_traffic_conversion.png")
    except Exception as e: print(f"  - Failed conversion plot: {e}")

    # 2. Regional Synchronization (18_regional_order_trend.png)
    try:
        geo = pd.read_csv(raw_dir / "geography.csv")
        orders = pd.read_csv(raw_dir / "orders.csv")
        customers = pd.read_csv(raw_dir / "customers.csv")
        
        # Join customers with geography on zip and city
        cust_geo = pd.merge(customers, geo, on=['zip', 'city'])
        order_geo = pd.merge(orders, cust_geo, on='customer_id')
        order_geo['order_date'] = pd.to_datetime(order_geo['order_date'])
        
        regional = order_geo.set_index('order_date').groupby([pd.Grouper(freq='ME'), 'region']).size().unstack().fillna(0)
        
        plt.figure(figsize=(10, 5))
        for region in regional.columns:
            plt.plot(regional.index, regional[region], label=region)
            
        plt.title('Geographic Synchronization of the 2019 Collapse', fontsize=14, fontweight='bold')
        plt.ylabel('Monthly Orders')
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / "18_regional_order_trend.png", dpi=300)
        plt.close()
        print("  - Saved 18_regional_order_trend.png")
    except Exception as e: print(f"  - Failed regional plot: {e}")

    # 3. Retention Collapse Heatmap (16_retention_heatmap.png)
    try:
        orders = pd.read_csv(raw_dir / "orders.csv")
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        orders['order_month'] = orders['order_date'].dt.to_period('Y') # Annual cohorts for clarity
        
        orders['cohort'] = orders.groupby('customer_id')['order_date'].transform('min').dt.to_period('Y')
        
        # Simple retention: % of cohort purchasing in subsequent years
        orders['years_since_first'] = (orders['order_date'].dt.year - orders['cohort'].dt.year)
        
        cohort_counts = orders.groupby(['cohort', 'years_since_first']).customer_id.nunique().reset_index()
        cohort_sizes = cohort_counts[cohort_counts['years_since_first'] == 0][['cohort', 'customer_id']].rename(columns={'customer_id': 'total'})
        
        retention = pd.merge(cohort_counts, cohort_sizes, on='cohort')
        retention['rate'] = (retention['customer_id'] / retention['total']) * 100
        
        pivot = retention.pivot(index='cohort', columns='years_since_first', values='rate')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot.iloc[:, :6], annot=True, fmt=".1f", cmap='YlGnBu')
        plt.title('Yearly Retention Decay Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(report_dir / "16_retention_heatmap.png", dpi=300)
        plt.close()
        print("  - Saved 16_retention_heatmap.png")
    except Exception as e: print(f"  - Failed retention plot: {e}")

if __name__ == "__main__":
    generate_eda_report_plots()
