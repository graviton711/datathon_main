import pandas as pd

def find_spikes():
    df = pd.read_csv('data/raw/sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate daily revenue
    daily = df.groupby('Date')['Revenue'].sum().reset_index()
    daily['Month'] = daily.Date.dt.month
    daily['Day'] = daily.Date.dt.day
    
    # Calculate monthly average to find relative spikes
    monthly_avg = daily.groupby([daily.Date.dt.year, daily.Date.dt.month])['Revenue'].transform('mean')
    daily['Ratio'] = daily['Revenue'] / monthly_avg
    
    # Find top 20 spikes in history
    spikes = daily.sort_values('Ratio', ascending=False).head(20)
    
    print("Top 20 Historical Spikes (Ratio vs Monthly Avg):")
    print(spikes[['Date', 'Revenue', 'Ratio']].to_string(index=False))
    
    # Check specific days like 6/6, 7/7, 11/11, 12/12
    print("\nEcommerce Day Performance (Avg Ratio):")
    for m in [6, 7, 9, 10, 11, 12]:
        day_ratio = daily[((daily.Month == m) & (daily.Day == m))].Ratio.mean()
        print(f" - {m}/{m}: {day_ratio:.2f}x")
    
    # Check Year-End Spikes (Dec 28-31)
    ye_ratio = daily[(daily.Month == 12) & (daily.Day >= 28)].Ratio.mean()
    print(f" - Year-End (Dec 28-31): {ye_ratio:.2f}x")

if __name__ == '__main__':
    find_spikes()
