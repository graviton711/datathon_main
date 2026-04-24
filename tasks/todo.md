# Task: Investigate Model Deviations

## Plan
- [x] Explore codebase to understand model architecture and evaluation process <!-- id: 0 -->
- [x] Locate evaluation results and identify days with largest deviations from best eval <!-- id: 1 -->
- [x] Experiment 1: Interaction Features for Peak Days (Failed to improve MAE) <!-- id: 6 -->
- [x] Experiment 2: Log/Sqrt Transformation for Target (Worsened MAE due to objective distortion) <!-- id: 7 -->
- [x] Experiment 3: Custom Loss Function (Tweedie) (Worsened MAE) <!-- id: 8 -->
- [x] Experiment 4: Time-Series Decomposition (Yearly Scale x Day Profile) (Worsened MAE due to stationarity loss) <!-- id: 10 -->
- [x] Experiment 5: Dynamic Peak Momentum (SUCCESS: Improved Weighted Rev MAE from 630k to 627k) <!-- id: 11 -->
- [x] Compare 5 methods and identify strengths/weaknesses <!-- id: 9 -->
- [x] Summarize findings and integrate the best solution (Peak Momentum) <!-- id: 5 -->

## Review
The investigation revealed that the model failed to capture growing peak day intensity. The best solution was a **Dynamic Peak Momentum** feature that carries forward the lift of the most recent major campaign. This improved Revenue MAE by 3,300 units and significantly reduced errors on major peak days (e.g., 31.03.2022).
