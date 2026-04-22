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