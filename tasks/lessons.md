# Lessons Learned & Pattern Repository

## Architecture & Configuration
- **Rule: Minimalist Config**: `src/config.py` should only contain infrastructure and core model hyperparameters. Business logic signals (lifts, momentum, seasonality boosts) must be derived dynamically from data in the pipeline to ensure regime-adaptability.
- **Rule: Fixed Submission Naming**: Production outputs must be named `submission.csv` to maintain consistency and avoid workspace clutter.

## Feature Engineering
- **Claude Signals**: 
    - Payday End (25-31) is a much stronger signal than Payday Start (1-5).
    - Quarter-end (last 7 days of Q) and Wednesdays are reliable high-confidence signals for revenue lift in the Vietnam retail market.
    - YoY Momentum should be calculated quarterly to capture seasonal growth shifts (e.g., Q1/Q3 recovery).

## Workflow
- **File Management**: Avoid creating multiple experiment files. Consolidate logic into the main `builder.py` and `pipeline.py` to maintain a single source of truth.

## Strategic Benchmarks
- **Current Best Benchmark**: **750k MAE** (Target for stability).
- **Ultimate Goal (Top 1)**: **610k MAE** (Target for winning).

## Modeling Philosophy
- **Model-Centric vs. Heuristics**: Avoid "code chay" (hardcoded multipliers). Use contextual flags and interaction terms.
- **Natural Scaling**: Scaling to 2024 must be grounded in fundamental drivers. 
- **Damped Momentum**: In volatile regimes (e.g. post-COVID), historical momentum should decay towards 1.0 (stability) over time. **CRITICAL**: Momentum must be compounded annually when forecasting multi-year horizons (e.g. 2024), otherwise growth will be lost.
- **Data-Driven Momentum**: Always ensure momentum calculations (like Q4 momentum) cover the period immediately preceding the forecast, even if it requires "virtual" year calculations based on recent training data.
- **Robust Anchoring**: Use a multi-year average for the base scale (e.g. 2-year median average) when the most recent year is a statistical anomaly (e.g. 2020) to ensure a stable starting point.

## Workflow & Coding Standards
- **Index Alignment**: ALWAYS use `.values` when assigning model predictions back to a slice of a DataFrame (e.g. `test_df['p'] = preds.values`) to prevent silent NaNs caused by index mismatch.
- **File Management**: Consolidate logic into `builder.py` and `pipeline.py`.

## Evaluation & Benchmarking
- **Score Discrepancy**: Local MAE is a relative indicator. Use CV for improvement tracking.
- **Current Performance**: Reached **653k Revenue MAE** using Damped Momentum on 2021-2022 validation.
