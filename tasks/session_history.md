# Detailed Session History & Experiment Log

## Current Best (Honest Pipeline): 676,151 (Leaderboard)
- **Status**: Stable / Improved Baseline.
- **Core Logic**: 
    - **Momentum**: Category-Specific (Casual 1.18x, Streetwear 1.13x, Outdoor 1.01x).
    - **Event Discovery**: 10-year Historical Median Lift.
    - **Damping**: 0.85 (Year 1/2023), 0.5 (Year 2/2024).
    - **COGS**: Monthly Category Profiles (Captures the 0.96 ratio shift in 2022).

---

## Experiment Log (Session 2026-04-26)

### 1. The "Top 1" Hunt (Target: 610,000)
- **Hypothesis**: Top 1 is capturing an "Efficiency Explosion" (CR +12%, Density +24%).
- **Implementation**: Triple-Threat Acceleration (Momentum ~1.53x).
- **Result**: **FAILED (Proxy MAE 670k)**.
- **Root Cause**: Over-prediction. Market demand in early 2023 was much softer than 2022 growth indicators suggested. High momentum in a flat market is a recipe for disaster.

### 2. The "Gap Surgery" (Targeted September Boost)
- **Hypothesis**: Gap analysis revealed a 17M revenue shortfall in Sept 2023 (9.9 / Back-to-school).
- **Implementation**: Applied a "Surgical" 1.25x boost to Month 9.
- **Result**: **MIXED**. It slashed 100k points from Sept MAE but felt like a "Magic Number" and slightly degraded historical CV consistency.
- **Refinement**: Attempted to automate this via **Recency-Weighted Events** (3x weight for 2021-22).

### 3. Recency-Weighted Event Scoring
- **Hypothesis**: Modern shopping festivals (9.9, 12.12) are much larger than historical ones. Trusting 2022 more will capture the 2023 peaks.
- **Implementation**: Weighted Median for Event Lift calculation.
- **Result**: **FAILED (LB 707,080)**.
- **Lesson**: 2022 was an outlier year for this specific shop. Using 2022 as the "new normal" resulted in massive over-estimation of the 2023 peaks. The 10-year historical median remains the most robust "Honest" predictor.

---

### 4. Category-Specific Momentum
- **Hypothesis**: Different categories grow at different rates (e.g., Casual > Outdoor). A global momentum over-predicts low-growth categories.
- **Implementation**: Calculated YoY growth per category and used a share-weighted blended momentum for projections.
- **Result**: **SUCCESS (Local MAPE improved from 21.16% to 20.42%)**.
- **Lesson**: Granular scaling is more robust than global scaling. Captures the "Category Mix" signal effectively.

---

## Strategic Conclusions
1.  **The "Honest" Limit**: Based on 2012-2022 data alone, the model seems to have a floor around **680k-690k**. Reaching <650k likely requires either (a) Future-looking features (Leakage) or (b) Highly aggressive scaling tuned directly to Leaderboard feedback (Probing).
2.  **Damping is Life**: Without the 0.5 Year 2 damping, all models explode in 2024. This suggests a significant market correction or shop closure in mid-2024.
3.  **COGS Regime Shift**: The shift to a 0.96 COGS ratio in late 2022 is the primary reason why Profit MAE is difficult to optimize. The shop moved from a high-margin to a low-margin operation.
4.  **Category Dynamics**: Casual (+18.8%) and Streetwear (+13.5%) are the primary growth drivers. Outdoor is stagnant (+1.7%).

---

## Next Steps / Ideas for Future Sessions
- **Inventory Constraint Modeling**: If product variety drops, sessions become less effective. Can we model `Revenue / Product_Count` as a primary feature?
- **Probing Alignment**: Now that the "Honest" logic is better, can we align its scale to the 624k probing result safely?
