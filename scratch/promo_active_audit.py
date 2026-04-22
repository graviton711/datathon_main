import pandas as pd

def audit_promo_active():
    promos = pd.read_parquet('data/processed/promotions.parquet')
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date'] = pd.to_datetime(promos['end_date'])
    
    results = []
    for yr in range(2018, 2023):
        for m in [2, 5]:
            test_date = pd.to_datetime(f"{yr}-{m:02d}-10") # Middle of the month
            mask = (promos['start_date'] <= test_date) & (promos['end_date'] >= test_date)
            count = promos[mask].shape[0]
            results.append({'Year': yr, 'Month': m, 'Active_Promo_Count': count})
            
    df = pd.DataFrame(results)
    print("=== ACTIVE PROMOTIONS AUDIT (Feb vs May) ===")
    print(df)

if __name__ == "__main__":
    audit_promo_active()
