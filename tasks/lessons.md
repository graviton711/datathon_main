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
- **Model-Centric vs. Heuristics**: Avoid "code chay" (hardcoded multipliers like `expected_market_lift *= 1.50`). Instead, provide the model with explicit contextual flags (`is_pre_tet`) and interaction terms (`tet_x_payday`). This allows the model to learn the true weights from the data distribution rather than relying on human-enforced overrides, ensuring better generalization to future data.
- **Natural Scaling**: Scaling to 2024 must be grounded in fundamental drivers (Traffic * CR * AOV). If a higher growth is needed to match the benchmark, it should be explained by the compounding of these drivers (e.g. CR improving over time) rather than arbitrary constants.
