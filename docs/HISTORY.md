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
- Start EDA in a notebook to answer Part 1 (MCQs) and Part 2 (Visualization) of the exam.
- Implement more detailed data loading logic in `src/data/loader.py` to handle the 15 CSV files.
- Establish a Time-Series validation strategy in `src/training/cross_val.py`.
