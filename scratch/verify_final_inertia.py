import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_final_inertia():
    # 1. Load Data
    sales = pd.read_parquet(DATA_DIR / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['year'] = sales['Date'].dt.year
    sales['month'] = sales['Date'].dt.month
    
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    orders['month'] = orders['Date'].dt.month
    
    df_items = pd.merge(items, orders[['order_id', 'year', 'month']], on='order_id')
    
    # 2. Aggregates
    annual_medians = sales.groupby('year')['Revenue'].median()
    
    # Q4 Signals
    q4_rev = sales[sales['month'] >= 10].groupby('year')['Revenue'].sum()
    q4_orders = orders[orders['month'] >= 10].groupby('year')['order_id'].count()
    q4_catalog = df_items[df_items['month'] >= 10].groupby('year')['product_id'].nunique()
    
    avail_years = sorted(list(set(q4_rev.index) & set(q4_orders.index) & set(q4_catalog.index) & set(annual_medians.index)))
    
    rows = []
    for i in range(2, len(avail_years)):
        yr_target = avail_years[i]
        yr_prev = avail_years[i-1]
        yr_prev2 = avail_years[i-2]
        
        # Target: Log Growth
        log_g = np.log(annual_medians[yr_target] / (annual_medians[yr_prev] + 1e-6) + 1e-6)
        
        # Signals: Q4 Momentum
        log_m_rev = np.log(q4_rev[yr_prev] / (q4_rev[yr_prev2] + 1e-6) + 1e-6)
        log_m_order = np.log(q4_orders[yr_prev] / (q4_orders[yr_prev2] + 1e-6) + 1e-6)
        log_m_aov = log_m_rev - log_m_order
        log_m_catalog = np.log(q4_catalog[yr_prev] / (q4_catalog[yr_prev2] + 1e-6) + 1e-6)
        
        rows.append({'y': log_g, 'x_rev': log_m_rev, 'x_order': log_m_order, 'x_aov': log_m_aov, 'x_catalog': log_m_catalog})
        
    train_df = pd.DataFrame(rows)
    
    # Baseline: Rev + Order + AOV
    X1 = sm.add_constant(train_df[['x_rev', 'x_order', 'x_aov']])
    model1 = sm.OLS(train_df['y'], X1).fit()
    
    # Final: Rev + Order + AOV + Catalog
    X2 = sm.add_constant(train_df[['x_rev', 'x_order', 'x_aov', 'x_catalog']])
    model2 = sm.OLS(train_df['y'], X2).fit()
    
    print("--- Final Inertia Model Verification ---")
    print(f"Scenario 1 (Current) R-squared: {model1.rsquared:.4f}")
    print(f"Scenario 2 (With Catalog) R-squared: {model2.rsquared:.4f}")
    print(f"Improvement: {(model2.rsquared - model1.rsquared):.4f}")
    
    print("\n--- Coefficients for Catalog Model ---")
    print(model2.params)
    print("\n--- P-values ---")
    print(model2.pvalues)

if __name__ == "__main__":
    verify_final_inertia()
