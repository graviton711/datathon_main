def _add_lags(df):
    df = df.copy()
    df['rev_lag_1'] = df['Revenue'].shift(1)
    df['rev_lag_7'] = df['Revenue'].shift(7)
    df['rev_roll_7'] = df['Revenue'].shift(1).rolling(7).mean()
    df['cogs_lag_1'] = df['COGS'].shift(1)
    df['cogs_lag_7'] = df['COGS'].shift(7)
    df = df.bfill()
    return df

print("Tested lags function.")
