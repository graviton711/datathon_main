import pandas as pd
import numpy as np

def add_time_features(df, date_col="Date"):
    """Adds basic and harmonic time-based features."""
    df = df.copy()
    
    # Basic
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Harmonics (Capture circularity of time)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Holidays
    df = add_holiday_features(df, date_col=date_col)
    
    return df

def add_holiday_features(df, date_col="Date"):
    """Adds Vietnamese public holiday indicators (Static for 2021-2024)."""
    df = df.copy()
    
    # 1. FIXED DATE HOLIDAYS
    # New Year (Jan 1)
    df['is_new_year'] = ((df[date_col].dt.month == 1) & (df[date_col].dt.day == 1)).astype(int)
    # Reunification (Apr 30)
    df['is_reunification'] = ((df[date_col].dt.month == 4) & (df[date_col].dt.day == 30)).astype(int)
    # Labor Day (May 1)
    df['is_labor_day'] = ((df[date_col].dt.month == 5) & (df[date_col].dt.day == 1)).astype(int)
    # National Day (Sep 2)
    df['is_national_day'] = ((df[date_col].dt.month == 9) & (df[date_col].dt.day == 2)).astype(int)
    
    # 2. LUNAR-BASED HOLIDAYS (Moving dates)
    # Tet (Lunar New Year) - Approx windows (+7 days from start)
    tet_dates = [
        ('2012-01-22', '2012-01-28'), ('2013-02-09', '2013-02-15'),
        ('2014-01-30', '2014-02-05'), ('2015-02-18', '2015-02-24'),
        ('2016-02-07', '2016-02-13'), ('2017-01-26', '2017-02-01'),
        ('2018-02-15', '2018-02-21'), ('2019-02-04', '2019-02-10'),
        ('2020-01-24', '2020-01-30'), ('2021-02-10', '2021-02-16'),
        ('2022-01-31', '2022-02-06'), ('2023-01-20', '2023-01-26'),
        ('2024-02-08', '2024-02-14')
    ]
    
    # Hung Kings Commemoration (10th of 3rd Lunar month)
    hung_kings = [
        '2012-03-31', '2013-04-19', '2014-04-09', '2015-04-28',
        '2016-04-16', '2017-04-06', '2018-04-25', '2019-04-14',
        '2020-04-02', '2021-04-21', '2022-04-10', '2023-04-29',
        '2024-04-18'
    ]
    
    df['is_tet'] = 0
    for start, end in tet_dates:
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        df.loc[mask, 'is_tet'] = 1
        
    df['is_hung_kings'] = df[date_col].dt.strftime('%Y-%m-%d').isin(hung_kings).astype(int)
    
    # Aggregate "Any Holiday"
    df['is_holiday'] = (
        df['is_new_year'] | df['is_reunification'] | 
        df['is_labor_day'] | df['is_national_day'] |
        df['is_tet'] | df['is_hung_kings']
    ).astype(int)
    
    return df
