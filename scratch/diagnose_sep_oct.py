"""
Investigate: what historical signals predict Sep/Oct revenue surge?
All analysis strictly on 2012-2022 data (Rule 14 compliant).
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('e:/VSCODE_WORKSPACE/NewDatathon')
sales = pd.read_parquet(ROOT / 'data/processed/sales.parquet')
sales['Date'] = pd.to_datetime(sales['Date'])
sales['year'] = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['day'] = sales['Date'].dt.day

# Use only 2019+ (post-regime-break)
sales_post = sales[sales['year'] >= 2019].copy()

# === 1. Monthly Lift vs Annual Mean (2012-2022) ===
print("=== 1. HISTORICAL MONTHLY LIFT (post-2019) ===")
annual_mean = sales_post.groupby('year')['Revenue'].mean()
monthly_mean = sales_post.groupby(['year','month'])['Revenue'].mean()

lift_df = []
for (yr, mo), rev_mean in monthly_mean.items():
    lift_df.append({'year': yr, 'month': mo, 'lift': rev_mean / annual_mean[yr]})
lift_df = pd.DataFrame(lift_df)

monthly_lift = lift_df.groupby('month')['lift'].agg(['mean','std','count'])
monthly_lift.columns = ['mean_lift','std','n_years']
print(monthly_lift.round(3).to_string())

# === 2. Sep and Oct specifically — year by year ===
print("\n=== 2. SEP/OCT LIFT BY YEAR (post-2019) ===")
for mo, mo_name in [(9, 'Sep'), (10, 'Oct')]:
    print(f"\n{mo_name}:")
    for yr in sorted(sales_post['year'].unique()):
        m_data = sales_post[(sales_post['year']==yr) & (sales_post['month']==mo)]
        full_data = sales_post[sales_post['year']==yr]
        if len(m_data) > 0:
            lift = m_data['Revenue'].mean() / full_data['Revenue'].mean()
            print(f"  {yr}: mean={m_data['Revenue'].mean():,.0f}  lift={lift:.3f}")

# === 3. Web traffic in Sep-Oct vs other months ===
print("\n=== 3. WEB TRAFFIC SEASONAL PATTERN (Sep/Oct vs avg) ===")
traffic = pd.read_parquet(ROOT / 'data/processed/web_traffic.parquet')
traffic['date'] = pd.to_datetime(traffic['date'])
traffic['year'] = traffic['date'].dt.year
traffic['month'] = traffic['date'].dt.month
traffic_post = traffic[traffic['year'] >= 2019]

daily_traffic = traffic_post.groupby(['year','month'])['sessions'].sum()
ann_traffic = traffic_post.groupby('year')['sessions'].sum()

traffic_lift = []
for (yr, mo), sess in daily_traffic.items():
    traffic_lift.append({'year':yr,'month':mo,'lift': sess / (ann_traffic[yr]/12)})
tl_df = pd.DataFrame(traffic_lift)
tl_monthly = tl_df.groupby('month')['lift'].mean()
print(tl_monthly.round(3).to_string())

# === 4. Q4 entry pattern: is Sep the ramp-up for Q4? ===
print("\n=== 4. Q4 RAMP-UP: Sep->Oct->Nov->Dec month-over-month ===")
for yr in sorted(sales_post['year'].unique()):
    row = []
    for mo in [8,9,10,11,12]:
        m_data = sales_post[(sales_post['year']==yr) & (sales_post['month']==mo)]
        row.append(m_data['Revenue'].mean() if len(m_data)>0 else np.nan)
    aug, sep, oct_, nov, dec = row
    print(f"{yr}:  Aug={aug:>12,.0f}  Sep={sep:>12,.0f}({sep/aug:.2f}x)  Oct={oct_:>12,.0f}({oct_/aug:.2f}x)")

# === 5. Order volume signal in Sep-Oct ===
print("\n=== 5. ORDER VOLUME MONTHLY LIFT (Sep/Oct post-2019) ===")
orders = pd.read_parquet(ROOT / 'data/processed/orders.parquet')[['order_id','order_date']]
orders['order_date'] = pd.to_datetime(orders['order_date'])
orders['year'] = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders_post = orders[orders['year'] >= 2019]

monthly_orders = orders_post.groupby(['year','month'])['order_id'].count()
ann_orders = orders_post.groupby('year')['order_id'].count()

for mo, mo_name in [(9,'Sep'),(10,'Oct')]:
    print(f"\n{mo_name} Order Lift:")
    for yr in sorted(orders_post['year'].unique()):
        if (yr, mo) in monthly_orders.index:
            lift = monthly_orders[yr,mo] / (ann_orders[yr]/12)
            print(f"  {yr}: orders={monthly_orders[yr,mo]:,}  lift={lift:.3f}")

# === 6. Correlation: prev Aug momentum -> Sep lift ===
print("\n=== 6. DOES PREV AUG REVENUE PREDICT SEP LIFT? ===")
aug_sep = []
for yr in sorted(sales_post['year'].unique()):
    aug_data = sales_post[(sales_post['year']==yr) & (sales_post['month']==8)]
    sep_data = sales_post[(sales_post['year']==yr) & (sales_post['month']==9)]
    prev_q2 = sales_post[(sales_post['year']==yr) & (sales_post['month'].isin([4,5,6]))]
    if len(aug_data)>0 and len(sep_data)>0:
        aug_sep.append({
            'year': yr,
            'aug_mean': aug_data['Revenue'].mean(),
            'sep_mean': sep_data['Revenue'].mean(),
            'sep_lift_vs_aug': sep_data['Revenue'].mean() / aug_data['Revenue'].mean(),
        })
aug_sep_df = pd.DataFrame(aug_sep)
print(aug_sep_df.to_string(index=False))
corr = aug_sep_df['aug_mean'].corr(aug_sep_df['sep_mean'])
print(f"\nCorr(Aug mean, Sep mean): {corr:.3f}")
