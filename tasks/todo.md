# Project Roadmap - Target 610k MAE

## Phase 1: Baseline & Validation Verification
- [x] Run `python src/evaluation/evaluate.py` to verify current score (1.19M MAE).
- [ ] Document detailed fold-wise performance in `HISTORY.md`.

## Phase 2: Signal & Model Refinement
- [ ] Feature Engineering: Investigate if `units_received` from inventory can be converted into a non-redundant signal (e.g., capacity usage %).
- [ ] Hyperparameter Tuning: Optimize LGBM parameters (`num_leaves`, `learning_rate`, `n_estimators`) specifically for normalized targets.
- [ ] Robustness: Implement outlier clipping for Revenue residuals during training.

## Phase 4: Ensembling & Final Submission
- [ ] Model Diversity: Add CatBoost or XGBoost to the `ForecastingPipeline`.
- [ ] Weighted Ensemble: Optimize weights between models using CV.
- [ ] Final Run: Generate `submissions/submission.csv` using full data training.
