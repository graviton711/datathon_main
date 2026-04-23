# Project Lessons & Patterns

## Architectural Patterns

### 1. Cyclic Time Encoding (Sin/Cos)
- **Problem**: Linear representation of days (1-31) creates a mathematical "cliff" between day 31 and day 1, making it hard to model end-of-month spending waves.
- **Solution**: Map days and months to a circle using Sine and Cosine transformations.
- **Result**: Improved MAE by ~10.5k by providing the model with a continuous representation of time.

### 2. The Recursive Stability Trap (Lag 364)
- **Problem**: Adding long-term lags like `rev_lag_364` can show massive improvement in 1-step validation (+23k MAE) but lead to catastrophic drift in 18-month recursive forecasting (-237k MAE).
- **Lesson**: NEVER trust a 1-step ahead validation score for a recursive pipeline. Always run a "Recursive Stress Test" before committing to a new feature.

### 3. Feature Contract Consistency
- **Problem**: Column order mismatch between `get_feature_names()` and `transform()` causes pipeline crashes during validation.
- **Fix**: Always ensure that the order in which columns are appended in `transform` matches the order in the list returned by `get_feature_names`.

### 4. The Direct Model Fallacy
- **Problem**: Attempting a simple "Direct" regression on raw values without normalization or growth calibration.
- **Lesson**: For multi-year datasets with significant market shifts (2020-2023), a naive direct model fails to capture scale changes. A recursive, stationary pipeline with an "Inertia" layer for scale projection is far superior for maintaining MAE stability across shifts.

## Data Insights (EDA)

### 1. Procurement vs. Revenue
- Found a **0.74 correlation** between `units_received` (Inventory Inbound) and Revenue. 
- **Pattern**: Retailers follow a strict "Procurement Cycle" (stocking up in April-June and December). This is a strong proxy for supply-side capacity.

### 2. Payday Elasticity Variance
- Payday lift is NOT constant across months. 
- **Pattern**: August (1.78x lift) and April (1.51x lift) are highly elastic, while November (1.11x) is suppressed due to mid-month 11.11 promo dilution.

## Debugging Lessons
- **Residual Audit**: Auditing the top 20 daily errors revealed that the model was failing most significantly at the end of the month (Payday window), leading to the discovery of the Cyclic Encoding and Payday Elasticity signals.
- **Redundant Static Features**: A strong domain correlation (like `units_received` vs Revenue at 0.88) is useless for Tree-based models (LightGBM) if the feature is static per month and `month` is already a categorical feature. The tree already splits on `month` to memorize seasonal means, so adding a 1-to-1 mapped monthly scalar adds zero information entropy and only introduces noise (ranked last in importance).

### 3. The Catalog Momentum Trap
- **Insight**: Adding SKU growth or Catalog volume as a growth signal (Inertia scaling) often introduces more noise than signal.
- **Lesson**: Growth in SKU count does not linearly translate to revenue growth in this dataset and can lead to biased/over-optimistic projections. Ignore this feature for the 610k target.

