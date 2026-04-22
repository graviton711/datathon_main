import pandas as pd

def audit_promo_seasonality():
    promos = pd.read_parquet('data/processed/promotions.parquet')
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date'] = pd.to_datetime(promos['end_date'])
    
    results = []
    for yr in range(2018, 2023):
        for m in [2, 5]: # Feb (Tet) vs May (Labor Day)
            mask = (promos['start_date'].dt.year == yr) & (promos['start_date'].dt.month == m)
            count = promos[mask].shape[0]
            avg_discount = promos[mask]['discount_value'].mean()
            results.append({'Year': yr, 'Month': m, 'Promo_Count': count, 'Avg_Discount': avg_discount})
            
    df = pd.DataFrame(results)
    print("=== PROMOTION SEASONALITY AUDIT (Feb vs May) ===")
    print(df)

if __name__ == "__main__":
    audit_promo_seasonality()
