import pandas as pd
import numpy as np
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("loader")

class DataLoader:
    def __init__(self):
        self.raw_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR

    def load_sales(self):
        """Loads the primary sales training data."""
        logger.info(f"Loading sales data from {config.SALES_TRAIN}")
        df = pd.read_csv(config.SALES_TRAIN, engine='pyarrow')
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def load_web_traffic(self):
        """Loads web traffic data for daily signals."""
        traffic_path = self.raw_dir / "web_traffic.csv"
        logger.info(f"Loading web traffic from {traffic_path}")
        df = pd.read_csv(traffic_path, engine='pyarrow')
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate by date (since source can have multiple records per day if stratified)
        # Assuming web_traffic is already daily but just in case:
        daily_traffic = df.groupby('date').agg({
            'sessions': 'sum',
            'unique_visitors': 'sum',
            'bounce_rate': 'mean'
        }).reset_index()
        daily_traffic.rename(columns={'date': 'Date'}, inplace=True)
        return daily_traffic

    def load_sentiment(self):
        """Loads and aggregates review sentiment to daily level."""
        reviews_path = self.raw_dir / "reviews.csv"
        logger.info(f"Loading reviews from {reviews_path}")
        df = pd.read_csv(reviews_path, engine='pyarrow')
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Aggregate to daily average rating
        daily_sentiment = df.groupby('review_date').agg({
            'rating': 'mean',
            'review_id': 'count'
        }).reset_index()
        daily_sentiment.rename(columns={'review_date': 'Date', 'rating': 'avg_rating', 'review_id': 'review_count'}, inplace=True)
        return daily_sentiment

    def get_merged_data(self):
        """Merges all data sources into a single modeling dataframe."""
        sales = self.load_sales()
        traffic = self.load_web_traffic()
        sentiment = self.load_sentiment()
        
        # Merge sources
        df = sales.merge(traffic, on='Date', how='left')
        df = df.merge(sentiment, on='Date', how='left')
        
        # Fill missing values for days with no traffic/reviews
        df['avg_rating'] = df['avg_rating'].ffill() # Carry forward sentiment
        df = df.fillna(0)
        
        logger.info(f"Merged modeling data shape: {df.shape}")
        
        # Reconciliation check
        total_revenue_original = sales['Revenue'].sum()
        total_revenue_merged = df['Revenue'].sum()
        
        if abs(total_revenue_original - total_revenue_merged) > 1e-5:
            logger.error("Revenue reconciliation failed during merge!")
        else:
            logger.info("Revenue reconciliation successful (100% matched).")
            
        return df

loader = DataLoader()
