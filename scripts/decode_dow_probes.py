"""
Decoder for Day-of-Week (DoW) Probing results.
Input the 7 MAE scores from LB to get exact actual Revenue sums and DoW multipliers.
"""
import pandas as pd
import numpy as np
import sys

# Constants from LB Probing Baseline
MAE_0 = 4183865.95
N = 1096  # 548 days * 2 targets
K = 10_000_000
TOTAL_REVENUE_GT = 4401156 * 548  # Global mean * days

# Horizon info
horizon_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
# DatetimeIndex doesn't need .dt
DOW_COUNTS = {i: (horizon_dates.dayofweek == i).sum() for i in range(7)}
DOW_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def decode(mae_list):
    print("=== LB DOW DECODING RESULTS ===")
    print(f"{'Day':<12} {'LB MAE':>12} {'Sum Actual (B)':>15} {'Mean Actual':>15} {'% of Total':>10}")
    print("-" * 70)
    
    total_decoded_rev = 0
    decoded_means = []
    
    for i, mae in enumerate(mae_list):
        n_dow = DOW_COUNTS[i]
        # Formula: Sum_Y = (MAE_0 + (n*K/N) - MAE_K) * (N/2)
        sum_y = (MAE_0 + (n_dow * K / N) - mae) * (N / 2)
        mean_y = sum_y / n_dow
        pct = (sum_y / TOTAL_REVENUE_GT) * 100
        
        total_decoded_rev += sum_y
        decoded_means.append(mean_y)
        
        print(f"{DOW_NAMES[i]:<12} {mae:>12,.2f} {sum_y/1e9:>15.3f}B {mean_y:>15,.0f} {pct:>9.2f}%")
    
    print("-" * 70)
    print(f"Total Decoded Revenue: {total_decoded_rev/1e9:.3f}B")
    print(f"Expected Revenue:      {TOTAL_REVENUE_GT/1e9:.3f}B")
    print(f"Reconciliation Error:  {(total_decoded_rev - TOTAL_REVENUE_GT)/1e6:.2f}M")
    
    # Calculate DoW Factors (Relative to Global Mean)
    global_mean = 4401156
    print("\n=== RECOMMENDED DOW MULTIPLIERS ===")
    for i, name in enumerate(DOW_NAMES):
        factor = decoded_means[i] / global_mean
        print(f"  {name:<12}: {factor:.4f}x")

if __name__ == "__main__":
    if len(sys.argv) == 8:
        results = [float(arg.replace(',', '')) for arg in sys.argv[1:]]
        decode(results)
    else:
        print("Usage: python decode_dow_probes.py MAE1 MAE2 ... MAE7")
        print("Order: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday")
