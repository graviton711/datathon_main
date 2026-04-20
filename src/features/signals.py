import pandas as pd
import numpy as np

def add_signal_features(df):
    """
    Adds lagged and rolling features based on Golden Lag analysis.
    - sessions/visitors: Lag 1 is the strongest short-term predictor.
    - sentiment: Rolling 30 is needed to capture business regime health.
    """
    df = df.copy()
    
    # Traffic Lags (Lead indicator for immediate sales)
    df['sessions_lag1'] = df['sessions'].shift(1)
    df['visitors_lag1'] = df['unique_visitors'].shift(1)
    
    # Traffic Momentum (Direction of growth)
    df['sessions_roll7'] = df['sessions'].rolling(window=7).mean()
    df['traffic_momentum'] = df['sessions'] / (df['sessions_roll7'] + 1e-6)
    
    # Sentiment Signals (Regime health)
    df['sentiment_roll30'] = df['avg_rating'].rolling(window=30).mean()
    
    # Sentiment Delta (Rate of trust decay/growth)
    df['sentiment_change_30d'] = df['sentiment_roll30'].diff(30)
    
    # Fill NAs caused by shifting/rolling
    df = df.ffill().fillna(0)
    
    return df

def add_blind_signals(df, frontier_date=None):
    """
    Adds 'Static Context Anchors' instead of daily lags.
    Useful for long-range forecasting where future daily traffic is unknown.
    
    Args:
        df: Merged dataframe.
        frontier_date: The last date of known truth. Everything after this uses frozen state.
    """
    df = df.copy()
    
    if frontier_date is None:
        frontier_date = df['Date'].max()
        
    # Calculate state at the frontier (using 90-day median to be robust)
    train_data = df[df['Date'] <= frontier_date]
    
    context_sessions = train_data['sessions'].tail(90).median()
    context_visitors = train_data['unique_visitors'].tail(90).median()
    context_sentiment = train_data['avg_rating'].tail(90).mean()
    
    # Broadcast these anchors to all rows (or only future rows if desired)
    # For a blind model, we use these as 'Current Scale' indicators
    df['anchor_sessions'] = context_sessions
    df['anchor_visitors'] = context_visitors
    df['anchor_sentiment'] = context_sentiment
    
    # Traffic Momentum at frontier
    # (How fast were we growing when we started the forecast?)
    last_7d = train_data['sessions'].tail(7).mean()
    last_30d = train_data['sessions'].tail(30).mean()
    df['anchor_momentum'] = last_7d / (last_30d + 1e-6)
    
    return df
