import pandas as pd

def get_vietnam_holidays(years):
    # Fixed dates
    fixed_holidays = [
        (1, 1, "New Year"),
        (2, 14, "Valentine"),
        (3, 8, "International Women's Day"),
        (4, 30, "Reunification Day"),
        (5, 1, "Labor Day"),
        (9, 2, "National Day"),
        (10, 20, "VN Women's Day"),
        (11, 20, "Teacher's Day"),
        (12, 24, "Christmas Eve"),
        (12, 25, "Christmas")
    ]
    
    # Lunar New Year (Tết) - First day of Lunar Calendar
    lunar_new_year_dates = {
        2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31',
        2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
        2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25',
        2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22',
        2024: '2024-02-10'
    }
    
    # Mid-Autumn (Rằm tháng 8) - Approximate
    mid_autumn_dates = {
        2012: '2012-09-30', 2013: '2013-09-19', 2014: '2014-09-08',
        2015: '2015-09-27', 2016: '2016-09-15', 2017: '2017-10-04',
        2018: '2018-09-24', 2019: '2019-09-13', 2020: '2020-10-01',
        2021: '2021-09-21', 2022: '2022-09-10', 2023: '2023-09-29',
        2024: '2024-09-17'
    }

    holidays = []
    for year in years:
        # Fixed
        for m, d, name in fixed_holidays:
            holidays.append({'Date': pd.Timestamp(f'{year}-{m:02d}-{d:02d}'), 'Holiday': name})
        
        # Tết
        if year in lunar_new_year_dates:
            holidays.append({'Date': pd.Timestamp(lunar_new_year_dates[year]), 'Holiday': 'Tet'})
            
        # Mid-Autumn
        if year in mid_autumn_dates:
            holidays.append({'Date': pd.Timestamp(mid_autumn_dates[year]), 'Holiday': 'Mid-Autumn'})
            
        # Retail Peaks (Double Days 1.1 to 12.12)
        for m in range(1, 13):
            holidays.append({'Date': pd.Timestamp(f'{year}-{m:02d}-{m:02d}'), 'Holiday': 'Retail Peak'})

    return pd.DataFrame(holidays).drop_duplicates(subset=['Date'])
