# Lessons Learned

## Feature Engineering
- **Peak Momentum Signal**: The strength of the most recent peak campaign is the strongest predictor for future peaks. Avoid using averages (mean/median) of multiple past peaks as it dilutes the signal in a growth regime.
- **Dynamic Feature Updates**: When implementing rolling or momentum features, ensure the logic is truly dynamic during the `fit` phase. If the feature is calculated as a constant from the entire training set, it provides zero information gain relative to the base feature (e.g., `event_score`).
- **Interaction Gating**: Momentum signals should be proportional to the inherent strength of the day (e.g., `last_peak_lift * event_score`) to avoid over-predicting normal days.
- **Biennial Strategic Patterns (Odd/Even Years)**: Some major campaigns (e.g., Urban Blowout) operate on a 2-year cycle. Using a global `is_odd_year` flag can introduce noise in 11 months of the year. A precise intersectional feature like `is_odd_year_aug` (August of Odd Years) is much more robust and effectively eliminated a 1.3M MAE COGS error in 2021.
- **Inference Clipping vs. Data Peaks**: A clipping threshold (e.g., q99) is a safety net. Only relax it if the model's raw predictions are being truncated. If the model predicts 1.36 and the actual is 1.44, the error is model confidence, not clipping. Keeping a tight clip at 1.41 prevents recursive error amplification.

## Debugging & Verification
- **Stationarity Check**: Always verify if the model is ignoring a new feature. If the results are identical to baseline, check for collinearity or constant values in the training set.
- **Data Leakage (Forensic)**: Be extremely careful when adding "lift" or "ratio" features. If they use the current day's target, they must be explicitly dropped before the model fit, even if the extractor handles other drops.
- **Recursive Consistency**: In recursive forecasting, any feature used by the model must be available or computable at every step of the future horizon.

## Workflow
- **Normalization/Scaling Consistency**: If training targets are normalized using a specific denominator (e.g., annual median), the inference pipeline MUST use the same logic for the base scale. Changing only the inference scale (e.g., to adjust for perceived level bias) without retraining with the same scale creates a "disconnect" that distorts predictions.
- **Strategic Probing**: When facing a significant performance gap (e.g., 99k) on an unseen horizon, use limited submission slots to isolate error sources (Revenue vs COGS) by keeping one component from the model and replacing the other with a naive baseline or constant.
- **Failed Feature Hypotheses (Promotions & DOM Profile)**: Explicitly encoding "manual shapes" (like a day-of-month median profile or binary promo flags) can backfire in recursive forecasting. GBDTs are already capable of learning these patterns from raw features (`day`, `month`). Adding redundant target-encoded features often introduces feedback loops that amplify errors in the recursive loop, leading to significantly worse MAE despite looking promising in static EDA.
- **Cleanliness**: Delete scratch scripts immediately after verification. Do not let `scratch/` become a graveyard of failed experiments.
- **Verification**: Always run the full multi-fold evaluation. A single fold (e.g., 2022 only) can be misleading due to regime shifts.

## Reporting & Visualization
- **Schema Verification**: Do not assume processed data follows the competition documentation exactly. Always verify column presence (e.g., `conversion_rate` was missing in `web_traffic.parquet` and required manual calculation from `orders`).
- **LaTeX Visualization**: Plotly is excellent for interactive EDA, but Matplotlib/Seaborn with 300 DPI and PDF/PNG export is necessary for professional LaTeX reports (NeurIPS style).
- **Data-Driven Narratives**: Use the diagnostic findings (e.g., the "Sizing Crisis") as the core of the report's prescriptive section to add business value beyond mere forecasting metrics.
