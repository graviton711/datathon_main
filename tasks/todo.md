# Task List - Revenue Forecasting Optimization

## 🔴 CRITICAL: Validation & Integrity (Highest Priority)
- [x] Implement 3-Fold Walk-Forward CV in `evaluate.py` (Weighted 20/30/50).
- [x] Eliminate `INERTIA_WEIGHT` magic number (0.8) by deriving it from historical correlation.
- [x] Replace `MOMENTUM_DAMPING` magic number (0.9) with data-driven decay based on YoY variance.
- [ ] Refine `_smooth_training_outliers` to be trend-aware instead of global-median-based.

## 🚀 Performance & Vectorization
- [ ] Vectorize `event_score` mapping in `builder.py` (Remove `.apply()`).
- [ ] Vectorize `prev_q4_momentum` mapping in `builder.py` (Remove `.apply()`).
- [ ] Centralize `is_signaled` logic into `BaselineFeatureExtractor` to ensure train/inference parity.

## 🧹 Code Quality & Maintenance
- [ ] Encapsulate data loading: Pass DataFrames into `fit`/`calibrate` instead of reading files inside Pipeline.
- [ ] Standardize logging output across all modules.
- [ ] Clean up `scratch/` directory files once insights are documented.

## ✅ Done
- [x] Vectorize `_get_days_to_tet` in `builder.py`.
- [x] Refactor recursive `predict` loop in `pipeline.py` to avoid $O(N^2)$ overhead.
- [x] Fix momentum compounding logic in `pipeline.py`.
- [x] Update `_calculate_q4_momentum_from_raw` to include latest year's impact.
- [x] Use `Config` paths in `_calculate_growth_calibration`.
- [x] Add safety check for empty `window_results` in momentum discovery.
- [x] Verify 2024 Tet lead-up features in `builder.py`.
- [x] Model COGS as a ratio of Revenue instead of absolute value.
- [x] Implement feature-contract validation between train/inference.
