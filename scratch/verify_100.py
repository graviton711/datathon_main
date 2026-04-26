import pandas as pd
import numpy as np

def get_mae_by_q(d1, d2):
    d1['Date'] = pd.to_datetime(d1['Date'])
    d2['Date'] = pd.to_datetime(d2['Date'])
    merged = pd.merge(d1, d2, on='Date', suffixes=('_1', '_2'))
    merged['AE'] = (np.abs(merged['Revenue_1'] - merged['Revenue_2']) + np.abs(merged['COGS_1'] - merged['COGS_2'])) / 2
    return merged.groupby(merged.Date.dt.to_period('Q'))['AE'].mean()

df_best = pd.read_csv('data/best_submit/best_624k.csv')
df_curr = pd.read_csv('submissions/submission.csv')

q_curr = get_mae_by_q(df_curr, df_best)

df_sim = df_curr.copy()
df_sim['Revenue'] = (df_sim['Revenue'] / 1.134) * 1.04
q_sim = get_mae_by_q(df_sim, df_best)

report = pd.DataFrame({
    'Current (1.13x)': q_curr,
    'Old Hack (1.04x)': q_sim
})
report['Improvement'] = report['Old Hack (1.04x)'] - report['Current (1.13x)']
print(report)
