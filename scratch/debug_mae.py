import pandas as pd
import sys
from pathlib import Path
PROJECT_ROOT = Path("e:/VSCODE_WORKSPACE/NewDatathon")
sys.path.append(str(PROJECT_ROOT))
from src.config import Config
from src.features.builder import BaselineFeatureExtractor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sales = pd.read_parquet(Config.SALES_TRAIN_FILE)
sales['Date'] = pd.to_datetime(sales['Date'])
train_sales = sales[sales['Date'].dt.year <= 2021].copy()
test_sales = sales[sales['Date'].dt.year == 2022].copy()

extractor = BaselineFeatureExtractor(date_col='Date')
extractor.fit(train_sales, y=train_sales['Revenue'])
X_train = extractor.transform(train_sales)
X_test = extractor.transform(test_sales)

model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
cats = [c for c in ['month', 'day_of_week', 'is_wednesday', 'is_weekend', 'is_payday_start', 'is_payday_end', 'is_quarter_end'] if c in X_train.columns]
model.fit(X_train, train_sales['Revenue'], categorical_feature=cats)
preds = model.predict(X_test)

print(f"2022 Actual Mean: {test_sales['Revenue'].mean():.0f}")
print(f"2022 Pred Mean  : {preds.mean():.0f}")
print(f"2022 Actual Std : {test_sales['Revenue'].std():.0f}")
print(f"2022 Pred Std   : {preds.std():.0f}")
print(f"MAE: {mean_absolute_error(test_sales['Revenue'], preds):.0f}")
