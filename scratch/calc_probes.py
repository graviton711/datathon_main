
import numpy as np

def calculate_mean_revenue(mae_probe, n_days, N=1096, mae_0=4183865.95, C=10000000):
    # Sum_Probe = nC - 2*Sum_Month + N*MAE_0
    # MAE_Probe = Sum_Probe / N = (nC/N) - (2*Sum_Month/N) + MAE_0
    # (2*Sum_Month/N) = (nC/N) + MAE_0 - MAE_Probe
    # Sum_Month = (nC + N*(MAE_0 - MAE_Probe)) / 2
    # Mean_Month = Sum_Month / n = (C/2) + (N/(2*n)) * (MAE_0 - MAE_Probe)
    
    mean_rev = (C/2) + (N/(2*n_days)) * (mae_0 - mae_probe)
    return mean_rev

# Months to calculate
probes = {
    "Nov 2023": {"mae": 4303887.08693, "days": 30},
    "Dec 2023": {"mae": 4346429.93536, "days": 31},
    "Feb 2024": {"mae": 4254376.48848, "days": 29},
    "Apr 2024": {"mae": 4118212.69773, "days": 30},
}

print("Calculated Mean Revenue for Probed Months:")
for month, data in probes.items():
    mean = calculate_mean_revenue(data["mae"], data["days"])
    print(f"{month}: {mean:,.2f}")

# Verify with Oct 2023 (known)
# Oct 23 Mean = 3,656,617
# Let's see what MAE would be
# 3656617 = 5000000 + (1096/(2*31)) * (4183865.95 - MAE_oct23)
# -1343383 = 17.677419 * (4183865.95 - MAE_oct23)
# -76000.0 = 4183865.95 - MAE_oct23
# MAE_oct23 = 4259865.95
oct23_mae = 4259865.95
print(f"Verification Oct 2023 (MAE {oct23_mae}): {calculate_mean_revenue(oct23_mae, 31):,.2f}")
