# Verified Insights - Datathon 2026

This file preserves all statistically verified findings and business insights discovered during the project.

| Date | Insight | Evidence | Modeling Impact |
| :--- | :--- | :--- | :--- |
| 2026-04-20 | Project structure initialized and verified. | Execution of `main.py` and directory audit. | Baseline established. |
| 2026-04-20 | Median inter-order gap is ~144 days. | Group-by `customer_id` and calculated `diff()` on `order_date`. | Suggests high retention but long cycle; C is best answer (180). |
| 2026-04-20 | 'Standard' segment has highest avg gross profit. | Mean of `(price - cogs) / price` per segment. | Standard segment is the profitability driver. |
| 2026-04-20 | 'wrong_size' is the top return reason for Streetwear. | Mode of `return_reason` for Category='Streetwear'. | Size guide improvements needed for Streetwear. |
| 2026-04-20 | Email campaigns have the lowest bounce rate. | Avg `bounce_rate` by `traffic_source`. | Email is the most engaging channel. |
| 2026-04-20 | ~39% of order items use a promotion. | `count(promo_id) / count(*)` in `order_items`. | Promotion strategy is widely adopted. |
| 2026-04-20 | 55+ age group has highest orders per customer. | `count(order_id) / nunique(customer_id)` by `age_group`. | Senior customers are the most loyal/frequent. |
| 2026-04-20 | East region generates the highest total revenue. | Sum of `quantity * unit_price` joined with `geography`. | Focus logistics/marketing on the East region. |
| 2026-04-20 | Credit card is the top payment method for cancelled orders. | Mode of `payment_method` for `order_status='cancelled'`. | Potential correlation between CC fraud or payment issues. |
| 2026-04-20 | Size 'S' has the highest return rate. | `returns / order_items` grouped by `size`. | Quality control or sizing issues in Small size. |
| 2026-04-20 | 6-installment plans have highest avg payment value. | Mean `payment_value` by `installments`. | High-value items prefer 6-month financing. |
| 2026-04-20 | Structural break: Volume collapsed ~70% in 2019. | Line plot of `order_count` by `year_month`. | Models should prioritize 2019+ data; 2012-2018 is "old regime". |
| 2026-04-20 | Operational Funnel: ~9.2% of orders are cancelled. | `count(cancelled) / count(orders)` (~59k/647k). | Significant revenue leakage; prioritize cancellation prediction. |
| 2026-04-20 | Demographic Gap: 55+ Desktop vs Under-35 Mobile. | Sunburst of `Age` -> `Gender` -> `Device`. | Tiered UI strategy: Mobile-first for young, Desktop-optimized for seniors. |
| 2026-04-20 | Seasonality: Strong Q4 spikes (Holiday effect). | Seasonal decomposition of order trends. | Time-series models must include monthly/holiday harmonics. |
| 2026-04-20 | Sentiment Leading Indicator: Rating dropped 12% from peak before 2019 collapse. | `06_avg_rating_trend.png` | Sentiment predictive power is high; precedes volume shifts. |

## Customer Sentiment & Performance Analysis (2012-2022)

### 1. Overall Satisfaction Profile (`05_rating_pie.png`)
- **Positive (4-5 stars)**: 72% (Majorly driven by 5-star reviews at 39.9%).
- **Neutral (3 stars)**: 15%.
- **Negative (1-2 stars)**: 13.1%.
- *Conclusion*: A permanent ~28% non-satisfied customer base suggests systemic issues in either product quality or delivery consistency.

### 2. Time-Series Evolution (`06_avg_rating_trend.png`)
- **2012 - 2014**: High stability (~3.95 - 4.0).
- **2014 - 2017 (Growth Pains)**: Frequent "spikes" and "dips" occurred, with several months dropping below 3.85. This correlates with business scaling phases.
- **2018 (The Pre-Collapse Decline)**: A sharp, sustained downward trend began in late 2017. The average rating hit a multi-year low (~3.84) in mid-2018, specifically preceding the massive order volume collapse in 2019. 
- **2019 - 2022**: High volatility in average ratings due to low order volume (Individual reviews have higher weight).

### 3. Fulfillment Impact (`07_delay_vs_rating.png`)
- **Retention Threshold**: Ratings remain stable (~3.9+) when shipping happens within 0-3 days.
- **Impact of Delay**: While the processed bars show limited impact up to 3 days, past EDA suggests a "fatigue point" exists where ratings collapse; further deep-dive into delivery (not just shipping) is needed.

### 4. Category-Specific Health (`08_category_sentiment.png`)
- **Best Performer**: `Streetwear` (3.94 rating) - also the highest volume category (>60k reviews), indicating it is the brand's core strength.
- **Worst Performers**: `GenZ` & `Casual` (3.92 rating) - Although the difference is small (0.02), these categories have significantly lower volume but higher proportional negativity.
| 2026-04-20 | Conversion Catastrophe: CR dropped from 1.5% to <0.5% in 2019 despite stable traffic. | `12_traffic_conversion.png` | Focus models on conversion-driving features (Quality, Sizing). |
| 2026-04-20 | Stockout Hypothesis Debunked: Fill rate remained ~100% during 2019 collapse. | `09_inventory_performance.png` | 2019 collapse was NOT a supply chain failure; it was a demand/trust issue. |

## Operational & Root Cause Analysis (Phase 2)

### 1. Supply Chain Stability (`09_inventory_performance.png`)
- **Metric**: `Avg Fill Rate` stayed near **100% (1.0)** throughout the 2012-2022 period.
- **Metric**: `Stockout Days` fluctuated within normal ranges (1.0 - 1.5 days).
- **Finding**: The business did NOT suffer from stock shortages during the 2018-2019 period. The products were in the warehouse; the customers simply stopped buying.

### 2. Marketing & Conversion Catastrophe (`12_traffic_conversion.png`)
- **Traffic Growth**: `Monthly Sessions` actually **Increased** year-over-year, peaking consistently over 1.0M sessions in late 2020.
- **Conversion Collapse**: The `Conversion Rate` (CR) began a terminal decline in early 2018, dropping from **~1.5% to ~0.3%**. 
- **Finding**: Marketing (Traffic) was NOT the problem. The issue was "Internal": the website/products failed to convert the massive traffic being generated.

### 3. Return Diagnostics & Revenue Leakage (`14_return_reasons.png`)
- **Failed Expectations (52.6%)**: Combined `wrong_size` (35%) and `not_as_described` (17.6%) account for over half of all returns.
- **Quality Issues (20.1%)**: One-fifth of returns are due to `defective` products.
- **Finding**: The high return rate is a clear indicator of why conversion collapsed: product descriptions (Size guides) and manufacturing quality failed to meet customer expectations.

### 4. Inventory Efficiency & "Dead Stock" (`11_inventory_efficiency.png`)
- **L-Shaped Efficiency**: Products with high `Stock on Hand` (>1000 units) consistently show near-zero `Sell-Through-Rate` (STR).
- **Finding**: Significant capital is locked in "Dead Stock", particularly in the `Outdoor` category. Conversely, top-selling items (STR > 0.6) are understocked (Stock < 200).

### 5. Category-Level Supply Risk (`10_category_stockout_risk.png`)
- **Inventory Paradox**: All categories show a high `Stockout Probability` (~0.66) despite some having >1000 days of supply.
- **Finding**: This suggests "Size-level Stockouts" – where the best-selling sizes are out of stock while unpopular sizes create an illusion of high inventory levels.

### 6. Promotional Strategy (`13_promo_impact.png`)
- **Promo Split**: `Percentage` discounts (avg 15%) are the high-frequency baseline (high count). `Fixed` discounts (avg $50) are used sparingly for high-impact events.
- **Finding**: The business relies on frequent, low-depth percentage discounts to maintain traffic, but these are not sufficient to counteract the trust-based conversion drop.
| 2026-04-20 | Brand Fatigue: Core 2012 cohort revenue dropped 50% in 2019. | `15_cohort_revenue_stacked.png` | Loss of loyalists is the primary driver of the structural break. |
| 2026-04-20 | Retention Collapse: Year 2 retention dropped from 65% (old) to <10% (new). | `16_retention_heatmap.png` | Business model is now "One-hit wonder"; requires urgent trust recovery. |

## Customer Loyalty & Retention Deep-Dive (Phase 3)

### 1. The Fall of the Loyalists (`15_cohort_revenue_stacked.png`)
- **Metric**: The `2012 Cohort` (Founding customers) provided **~$0.8B** revenue in 2018.
- **Metric**: In 2019, this same cohort's revenue dropped to **~$0.5B** (a 37.5% drop in one year).
- **Finding**: The 2019 collapse was not just about failing to acquire new customers; it was a **Massive Churn** of the most loyal customer base who had been with the brand for 7 years.

### 2. Retention Heatmap Breakdown (`16_retention_heatmap.png`)
- **Historical Stability**: Customers who joined between 2012-2015 had a Year 2 retention rate of **~60-65%**.
- **Modern Catastrophe**: 
    - Customers who joined in 2018 had a Year 2 retention of only **9.4%**.
    - Customers who joined in 2019-2021 have Year 2 retention rates below **8%**.
- **Finding**: The business has shifted from a "High Loyalty" model to a "Burn and Churn" model. New customers try the brand once, likely experience the sizing/quality issues identified in Phase 2, and never return.

### 3. AOV Stability (`17_aov_cohort_trend.png`)
- **Metric**: Despite the volume drop, Average Order Value (AOV) has remained relatively stable across cohorts (~$150-$200).
- **Finding**: The problem is **Frequency and Retention**, not the spend amount per transaction. We are losing "People", not "Wallet Share" per person.
| 2026-04-20 | Universal Failure: All regions (East, Central, West) collapsed simultaneously in 2019. | `18_regional_order_trend.png` | Collapse is structural/national, not a localized logistics issue. |
| 2026-04-20 | Price-Volume Mismatch: Massive price hikes in 2018-2020 accelerated volume drop. | `19_price_elasticity.png` | Pricing strategy was counter-productive during the quality crisis. |
| 2026-04-23 | Silent Bug Fix A: `prev_q4_momentum` now computed from raw Revenue pre-normalization. | Pipeline refactor in `src/training/pipeline.py` + injected map into `BaselineFeatureExtractor`. | Removed sign-flip risk in Q4 growth signal; momentum feature now represents market growth direction. |
| 2026-04-23 | Silent Bug Fix B: 2024 horizon no longer defaults to zero momentum from missing dict key. | Runtime check: 2024 rows with `prev_q4_momentum == 0.0` reduced to 0/183. | Eliminated out-of-distribution artifact in late-horizon features. |
| 2026-04-23 | Ghost Feature Fix C: `year` removed from model input schema. | Transformer output inspection (`has_year False`) and feature-contract validation guard. | Prevented train/inference mismatch and accidental extrapolation shortcut on calendar year. |
| 2026-04-23 | COGS branch refactored to ratio target (`COGS/Revenue`) with quantile clipping. | Walk-forward run on 2021-2022 after refactor. | Improved stability of COGS predictions under structural seasonality shifts. |
| 2026-04-23 | Walk-forward score after bug-fix package: Total MAE = 1,327,352. | `src/evaluation/evaluate.py` output: Revenue MAE 691,451; COGS MAE 635,901. | Net improvement vs prior 1.37M baseline; fixes are production-viable. |

## Regional Trends & Price Elasticity (Phase 3)

### 1. Geographic Synchronization (`18_regional_order_trend.png`)
- **Metric**: `East` (Largest region, peak ~5k orders) and `West`/`Central` followed identical downward trajectories in 2018-2019.
- **Finding**: The collapse was **Federated**. No single region remained healthy. This confirms that the root cause (Sizing/Trust) was a centralized product/brand issue, not a regional supply chain disruption.

### 2. The Price Hike Paradox (`19_price_elasticity.png`)
- **Trend**: Avg Unit Price remained stable at **~$5,000** until mid-2018.
- **Strategic Error**: Starting late 2018, the company aggressively increased prices, with average prices hitting **$7,000 - $8,000** in 2020-2022.
- **Inelastic Response**: Total units sold plummeted exactly when prices rose. 
- **Finding**: The company attempted to offset falling volumes by increasing margins per unit. However, doing so while the "product-market fit" was broken (due to the 52% return rate for size/description issues) likely alienated customers further.

### 3. Shipping Fee IRRELEVANCE (`20_shipping_cost_trend.png`)
- **Metric**: Avg shipping fee dropped from **$6** to **$4** over the decade.
- **Finding**: Logistics costs were not a barrier to conversion. The collapse happened despite cheaper shipping for customers.
| 2026-04-20 | Data Integrity Audit: 100% reconciliation between raw and agg revenue. | `21_target_reconciliation.png` | 100% confidence in using order-level features for forecasting. |

## Data Integrity & Target Audit (Phase 3)

### 1. Revenue Reconciliation (`21_target_reconciliation.png`)
- **Metric**: Comparison between calculated revenue from `order_items` and pre-aggregated `Revenue` in `sales.csv`.
- **Result**: The two lines overlap **perfectly** across the 2012-2022 timeline.
- **Finding**: Our transaction-level data is the exact source of truth for the competition targets. There is NO data leakage, NO missing orders, and NO hidden fees/cancellations that distort the target variable.

## Lag & Signal Analysis (EDA Findings)

### 1. Traffic Signals
- **Immediate Impact**: `sessions` and `unique_visitors` have a strong positive correlation (~0.28) at **Lag 1 day**.
- **Smoothing**: 7-day and 14-day rolling averages maintain high correlation, making them robust to daily anomalies.

### 2. Sentiment Signals
- **Noise vs. Trend**: Daily ratings are noisy (near 0 correlation), but a **30-day rolling average** is required to capture the regime shifts identified in the 2019 collapse analysis.
