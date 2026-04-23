import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("e:/VSCODE_WORKSPACE/NewDatathon/data/processed")

def analyze_device_payment():
    # 1. Load Data
    orders = pd.read_parquet(DATA_DIR / "orders.parquet")[['order_id', 'order_date', 'payment_method', 'device_type']]
    orders['Date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['Date'].dt.year
    
    # Load items to get AOV
    items = pd.read_parquet(DATA_DIR / "order_items.parquet")
    items['item_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
    order_rev = items.groupby('order_id')['item_rev'].sum().reset_index()
    
    df = pd.merge(orders, order_rev, on='order_id', how='left')
    
    # 2. Analyze Device Mix Over Time
    device_share = df.groupby(['year', 'device_type']).size().unstack(fill_value=0)
    device_share = device_share.div(device_share.sum(axis=1), axis=0)
    
    print("--- Device Share Over Years ---")
    print(device_share)
    
    # 3. Analyze Payment Mix Over Time
    payment_share = df.groupby(['year', 'payment_method']).size().unstack(fill_value=0)
    payment_share = payment_share.div(payment_share.sum(axis=1), axis=0)
    
    print("\n--- Payment Share Over Years ---")
    print(payment_share)
    
    # 4. AOV per Device/Payment in the latest year (2022)
    df_2022 = df[df['year'] == 2022]
    aov_device = df_2022.groupby('device_type')['item_rev'].mean()
    aov_payment = df_2022.groupby('payment_method')['item_rev'].mean()
    
    print("\n--- AOV in 2022 by Category ---")
    print("By Device:\n", aov_device)
    print("\nBy Payment:\n", aov_payment)
    
    # 5. Check for Structural Shifts
    # Is "Mobile" AOV significantly different from "Desktop"?
    # Is "Credit Card" AOV higher than others?
    
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    device_share.plot(kind='area', stacked=True, ax=plt.gca(), alpha=0.7)
    plt.title("Device Mix Shift")
    
    plt.subplot(1, 2, 2)
    payment_share.plot(kind='area', stacked=True, ax=plt.gca(), alpha=0.7)
    plt.title("Payment Mix Shift")
    
    plt.tight_layout()
    plt.savefig("e:/VSCODE_WORKSPACE/NewDatathon/data/plots/device_payment_shift.png")

if __name__ == "__main__":
    analyze_device_payment()
