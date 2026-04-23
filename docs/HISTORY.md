# Project History - Datathon 2026

This file tracks the evolution of the project and serves as context for future sessions.

## [2026-04-20] Session 1: Project Initialization

### Tasks Accomplished:
- **Project Structure**: Created a professional modular layout (`src/`, `data/`, `models/`, `submissions/`, `logs/`).
- **Core Modules**: Initialized `config.py`, `constants.py`, `loader.py`, `lgbm_model.py`, and `logger.py`.
- **Environment**: Created a virtual environment (`venv`) and installed all core ML dependencies (`pandas`, `LightGBM`, `CatBoost`, `XGBoost`, `scikit-learn`, etc.).
- **Rules & Governance**: Established `docs/PROJECT_RULES.md` and this `HISTORY.md` file for session persistence.
- **Verification**: Successfully ran a dry-run of the pipeline via `main.py`.

### Key Decisions:
- Used **Python classes** for configuration instead of YAML for tighter integration with the codebase.
- Chose **LightGBM** as the default baseline model wrapper.
- Structured feature engineering into sub-modules (temporal, categorical, aggregations) for cleaner experimental cycles.

### Current State:
- All core folders and files are created and importable.
- Virtual environment is fully populated.
- Raw data exists in `data/raw/` (verified via structures).
- Ready for Exploratory Data Analysis (EDA) and baseline modeling.

### Next Steps:
- Continue EDA for other tables (Products, Promotions, Web Traffic) to complete Part 2.
- Integrate the verified insights into the technical report.
- Develop the `src/data/loader.py` to support full pipeline training.

## [2026-04-20] Session 2: Analysis & Visualization

### Tasks Accomplished:
- **MCQ Resolution**: Successfully solved all 10 problems in Part 1 (MCQ) using `scripts/mcq_engine.py`. Answers verified against provided options.
- **Interactive EDA**: Created `notebooks/01_orders_eda.ipynb` using **Plotly**.
- **Cross-Table Insights**: Uncovered relationships between `Orders`, `Customers`, and `Revenue`.
- **Insight Logging**: Standardized the logging of findings in `docs/VERIFIED_INSIGHTS.md`.

### Key Decisions:
- Adopted **Plotly** as the primary visualization library for its interactive capabilities.
- Defined a **4-level analysis framework** (Descriptive to Prescriptive) within the notebooks to align with exam criteria.

### Current State:
- Part 1 (MCQ) is effectively complete.
- Part 2 (EDA) is underway with the transactional layer analyzed.
- Verified insights are logged and ready for reporting.

### Next Steps:
- Perform product-level EDA to analyze category profitability.
- Analyze promotion effectiveness and seasonality in detail.
- Establish the official baseline model (Naive Baseline) to meet competition requirements.

## [2026-04-20] Session 3: Diagnostic EDA & Root Cause Identification

### Tasks Accomplished:
- **Phase 1 (Reviews & Sentiment)**: Analyzed 2012-2022 reviews. Discovered that **Average Rating** is a leading indicator (dropped 12% before the 2019 volume collapse).
- **Phase 2 (Operational Funnel)**: Debunked the "Supply Chain Failure" hypothesis (Fill Rate remained ~100%).
- **Conversion Deep-Dive**: Identified **Conversion Rate (CR)** collapse from 1.5% to 0.3%. Linked it to **Sizing Issues** (>35% of returns due to `wrong_size`).
- **Inventory Audit**: Discovered the "Inventory Paradox" (High stock on hand in total, but frequent size-level stockouts for best-sellers).

### Key Decisions:
- Shifted analytical focus from "Logistics" to "Customer Trust & Product Quality" as the primary driver for forecasting.
- Enforced a **Notebook-only EDA workflow** to keep the project structure clean and reproducible.

## [2026-04-20] Session 4: Strategy, Retention & Data Audit

### Tasks Accomplished:
- **Phase 3 (Cohort Analysis)**: Confirmed the 2019 collapse was a **Veteran Churn** event (2012 loyalist revenue dropped 50%).
- **Retention Audit**: Found new customer retention dropped from **~65%** (pre-2017) to **<10%** (2019+).
- **Economic Analysis**: Identified a "Price Hike Mismatch" where unit prices rose to $7,000+ while volume was already in freefall.
- **Geographic Check**: Verified the 2019 collapse was **National (Universal)**, not regional.
- **Audit (Success)**: Achieved **100% reconciliation** between transactional revenue and aggregated sales reports.

### Key Decisions:
- Confirmed that transaction-level data (`order_items`) is the "Source of Truth" for all competition targets.
- Decided to implement **Regime-Aware modeling** (partitioning logic or sample weighting) for the forecasting phase to account for the 2019 structural break.

### Current State:
- EDA phase is **100% Complete**.
- Root Cause is fully documented and statistically verified.
- Data integrity is confirmed for modeling.

## [2026-04-23] Session 5: Model Refinement & Stabilization

### Tasks Accomplished:
- **Comprehensive Audit**: Reviewed all `docs/`, `src/`, and `data/` directories.
- **Bug Fix**: Resolved a critical index alignment bug in `evaluate.py` that caused NaNs in metrics.
- **Feature Enrichment**: Integrated high-confidence signals (Wednesdays, Payday End, Quarter End) into the pipeline.
- **Momentum Stabilization**: Implemented **Damped Dual-Momentum** and **Robust Base Scale** (2-year average) to handle regime shifts.
- **Performance Leap**: Reduced Revenue MAE from **891k** to **653k** (Total MAE 1.37M) in Walk-Forward validation.

### Key Decisions:
- Adopted **Micro-average ratios** (weighted sums) for momentum calculation to ensure statistical stability.
- Shifted from "Exponential Momentum" to "Damped Momentum" to prevent unrealistic long-term projections.

### Current State:
- Project structure and logic are well-understood.
- Modeling pipeline is stable and ready for further optimization.

### Next Steps:
- Proceed with instructions from the user for further model refinement or reporting.

## [2026-04-23] Session 6: Silent-Bug Fixes & COGS Ratio Refactor

### Tasks Accomplished:
- **Bug A Fixed (Q4 Momentum Source)**: Moved `prev_q4_momentum` source from normalized training target to a raw-revenue Q4 momentum map computed before normalization.
- **Bug B Fixed (2024 Zero Artifact)**: Added robust fallback logic for unseen years (carry-forward latest known momentum), eliminating missing-key `0.0` leakage in 2024.
- **Bug C Fixed (Ghost Feature)**: Removed `year` from transformer output to prevent unintended model leakage and train/inference feature drift.
- **COGS Refactor**: Replaced absolute COGS forecasting branch with `COGS/Revenue` ratio modeling, then reconstructed final COGS as `final_revenue * predicted_ratio`.
- **Schema Guardrails**: Added strict feature-contract validation to enforce identical transformed columns between training and inference.
- **Validation Run**: Executed walk-forward evaluation (`Train<=2020`, `Test=2021-2022`) successfully after refactor.

### Key Decisions:
- Keep Revenue as the primary absolute forecast target, and model COGS as a dependent ratio to improve stability under regime shifts.
- Use quantile clipping on predicted ratio (derived from training target distribution) to reduce extreme COGS forecasts.
- Enforce deterministic transformer output ordering (`lag features + declared engineered features`) to avoid silent feature drift.

### Current State:
- Silent bugs identified by feature audit are patched in the production pipeline.
- Feature schema is now explicit and validated.
- Walk-forward score improved to **Total MAE: 1,327,352** with Revenue MAE **691,451** and COGS MAE **635,901**.

### Next Steps:
- Run ablation to isolate gain contribution from each fix (A/B/C vs ratio refactor).
- Tune COGS ratio clipping bounds and momentum decay jointly for further MAE reduction.

## [2026-04-23] Session 7: Comprehensive Code Review & Roadmap Update

### Tasks Accomplished:
- **Full Code Audit**: Reviewed `config.py`, `builder.py`, `pipeline.py`, and `evaluate.py` against Project Rules.
- **Gap Identification**: Identified missing 3-Fold CV, hardcoded "magic numbers" (`INERTIA_WEIGHT`, `DAMPING`), and performance bottlenecks (`.apply()`).
- **Roadmap Refactoring**: Updated `tasks/todo.md` with a prioritized list focusing on validation integrity and data-driven parameters.

### Key Decisions:
- Prioritize **Validation Integrity** (3-Fold CV) as the next critical step to avoid overfitting and ensure path to 610k.
- Aggressively remove remaining magic numbers in `Config` by making them derived from training data statistics.

### Current State:
- Project roadmap is now strictly aligned with 610k target.
- Technical debt (vectorization, magic numbers) is mapped and ready for resolution.

### Tasks Accomplished (Update):
- **Implemented 3-Fold Walk-Forward CV**: Upgraded `evaluate.py` to support weighted 3-fold validation (20/30/50 split). 
    - Results: Mean Total MAE = 1.48M, Weighted Total MAE = 1.41M. 
    - Insight: Revenue MAE remains stable at ~700k for 2021-2022, but 2020 (1.3M) remains a significant outlier.

### Next Steps:
- **Eliminated Magic Numbers**: Refactored `pipeline.py` to derive `INERTIA_WEIGHT` and `MOMENTUM_DAMPING` from historical data statistics and R-squared confidence. 
    - Result: Successfully fulfilled Rule 10 by removing hardcoded 0.8 and 0.9 multipliers from `Config`.
- Refine outlier smoothing to be trend-aware.

## [2026-04-23] Session 8: Stationary Lag Normalization & Recursive Stabilization

### Tasks Accomplished:
- **Year-Boundary Bias Fix**: Identified and resolved a critical scale-mismatch in lag features. Lags are now calculated on absolute values and normalized dynamically by the target year's scale during both training and inference.
- **Pipeline Refactor**: Switched `ForecastingPipeline` buffer to store absolute revenue, ensuring recursive predictions use consistent stationary lags.
- **Data-Driven Weights**: Verified the impact of anomaly weighting based on the Revenue/Sessions efficiency ratio.

### Key Decisions:
- Mantained the **Recursive Pipeline** but hardened it against "Snowball effect" by ensuring lag features are always stationary relative to the current year's projected median.
- Chose **Dynamic Normalization** over static per-year normalization to remove the "cliff" artifact at Jan 1st.

### Current State:
- Pipeline is structurally sound and mathematically consistent across year boundaries.
- Ready for full 3-Fold Walk-Forward validation to verify score improvement toward the 610k target.

## [2026-04-23] Session 9: Context Synchronization & Baseline Verification

### Tasks Accomplished:
- **Full Project Audit**: Thoroughly reviewed all `src/`, `docs/`, and `tasks/` files to synchronize with the current project state.
- **Baseline Verification**: Executed `evaluate.py` to establish a clean starting point for the new session.
- **Roadmap Creation**: Created `tasks/todo.md` to track progress toward the 610k MAE target.

### Key Decisions:
- Verified that the current **Weighted Total MAE is 1,190,251** (Revenue 631k, COGS 560k). This is significantly better than the previous 1.32M record, confirming the efficacy of the Session 8 refactors.
- Confirmed "Catalog Momentum" as the next high-priority optimization.

### Current State:
- Pipeline is verified and performing at 1.19M Total MAE.
- All documentation is up-to-date and reflects the current verified score.
- Ready to implement Catalog Momentum.

### Next Steps:
- Implement Catalog Momentum by extracting SKU counts and integrating them into the inertia calibration logic.
