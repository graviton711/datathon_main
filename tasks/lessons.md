# Lessons Learned

## Feature Engineering
- **Peak Momentum Signal**: The strength of the most recent peak campaign is the strongest predictor for future peaks. Avoid using averages (mean/median) of multiple past peaks as it dilutes the signal in a growth regime.
- **Dynamic Feature Updates**: When implementing rolling or momentum features, ensure the logic is truly dynamic during the `fit` phase. If the feature is calculated as a constant from the entire training set, it provides zero information gain relative to the base feature (e.g., `event_score`).
- **Interaction Gating**: Momentum signals should be proportional to the inherent strength of the day (e.g., `last_peak_lift * event_score`) to avoid over-predicting normal days.

## Debugging & Verification
- **Stationarity Check**: Always verify if the model is ignoring a new feature. If the results are identical to baseline, check for collinearity or constant values in the training set.
- **Data Leakage (Forensic)**: Be extremely careful when adding "lift" or "ratio" features. If they use the current day's target, they must be explicitly dropped before the model fit, even if the extractor handles other drops.
- **Recursive Consistency**: In recursive forecasting, any feature used by the model must be available or computable at every step of the future horizon.

## Workflow
- **Cleanliness**: Delete scratch scripts immediately after verification. Do not let `scratch/` become a graveyard of failed experiments.
- **Verification**: Always run the full multi-fold evaluation. A single fold (e.g., 2022 only) can be misleading due to regime shifts.
