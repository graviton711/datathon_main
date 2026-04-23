import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def verify_inertia_with_rating():
    # 1. Load Data
    df = pd.read_parquet(DATA_DIR / "sales.parquet")
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    
    reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
    reviews['review_date'] = pd.to_datetime(reviews['review_date'])
    reviews['year'] = reviews['review_date'].dt.year
    reviews['month'] = reviews['review_date'].dt.month
    
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['order_date'].dt.year
    orders['month'] = orders['order_date'].dt.month
    
    # 2. Aggregates
    annual_medians = df.groupby('year')['Revenue'].median()
    
    # Q4 Signals per year
    q4_rev = df[df['month'] >= 10].groupby('year')['Revenue'].sum()
    q4_orders = orders[orders['month'] >= 10].groupby('year')['order_id'].count()
    q4_rating = reviews[reviews['month'] >= 10].groupby('year')['rating'].mean()
    
    avail_years = sorted(list(set(q4_rev.index) & set(q4_orders.index) & set(q4_rating.index) & set(annual_medians.index)))
    
    rows = []
    for i in range(2, len(avail_years)):
        yr_target = avail_years[i]
        yr_prev = avail_years[i-1]
        yr_prev2 = avail_years[i-2]
        
        # Target: Log of realized annual growth
        log_g = np.log(annual_medians[yr_target] / (annual_medians[yr_prev] + 1e-6) + 1e-6)
        
        # Signals: Log of Q4 YoY Momentum
        log_m_rev = np.log(q4_rev[yr_prev] / (q4_rev[yr_prev2] + 1e-6) + 1e-6)
        log_m_order = np.log(q4_orders[yr_prev] / (q4_orders[yr_prev2] + 1e-6) + 1e-6)
        log_m_aov = log_m_rev - log_m_order
        log_m_rating = np.log(q4_rating[yr_prev] / (q4_rating[yr_prev2] + 1e-6) + 1e-6)
        
        rows.append({
            'y': log_g, 
            'x_rev': log_m_rev, 
            'x_order': log_m_order, 
            'x_aov': log_m_aov, 
            'x_rating': log_m_rating
        })
        
    train_df = pd.DataFrame(rows)
    
    # Kịch bản 1: Baseline
    X1 = sm.add_constant(train_df[['x_rev', 'x_order', 'x_aov']])
    model1 = sm.OLS(train_df['y'], X1).fit()
    
    # Kịch bản 2: With Rating
    X2 = sm.add_constant(train_df[['x_rev', 'x_order', 'x_aov', 'x_rating']])
    model2 = sm.OLS(train_df['y'], X2).fit()
    
    print("--- Inertia Regression Verification ---")
    print(f"Scenario 1 (Current) R-squared: {model1.rsquared:.4f}")
    print(f"Scenario 2 (With Rating) R-squared: {model2.rsquared:.4f}")
    print(f"\nImprovement: {(model2.rsquared - model1.rsquared):.4f}")
    
    print("\n--- Coefficients for Rating Scenario ---")
    print(model2.params)
    
    # P-values to see significance
    print("\n--- P-values ---")
    print(model2.pvalues)

if __name__ == "__main__":
    verify_inertia_with_rating()
