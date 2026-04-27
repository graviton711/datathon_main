# Project Working Rules - Datathon 2026

To ensure consistency, reproducibility, and smooth transitions between sessions, the following rules MUST be followed:

## 1. History Tracking (Mandatory)
- **Always** maintain and update the [HISTORY.md](file:///e:/VSCODE_WORKSPACE/NewDatathon/docs/HISTORY.md) file at the end of every session or major milestone.
- This file serves as the "memory" for the AI assistant across different chat sessions.
- Each entry should include: Date, Task Accomplished, Key Decisions, and Current State.

## 2. Resource Awareness
- **Read Documentation First**: Before implementing any feature, check the `docs/` folder for rules, schemas, and requirements (e.g., `EXAM_V1.md`).
- **Data Integrity**: Always refer to the data schemas in `docs/` and verify data presence in `data/raw/` before loading.

## 3. Environment Discipline
- **Use Virtual Environment (`venv`)**: Every command (pip, python, etc.) must be run using the project's virtual environment located at `e:/VSCODE_WORKSPACE/NewDatathon/venv`.
- **Requirements**: If a new library is needed, install it in the venv and update `requirements.txt` immediately.

## 4. Coding Standards
- Maintain the modular structure in `src/`.
- Use `src/config.py` for all paths and global settings.
- Never hardcode magic numbers or local paths outside of config.

## 5. Verified Insights Tracking
- **Insight Log**: Every statistically verified finding, critical pattern, or confirmed hypothesis must be documented in [VERIFIED_INSIGHTS.md](file:///e:/VSCODE_WORKSPACE/NewDatathon/docs/VERIFIED_INSIGHTS.md).
- **Structure**: Each insight should state the finding, the supporting evidence (e.g., notebook reference or metric), and its impact on the modeling strategy.
- This ensures that our final report is easy to compile and based on solid evidence.

## 6. Git Version Control & Commit Strategy
- **Baseline Commit**: Initial project structure and venv setup must be committed immediately.
- **Score Improvement**: A Git commit MUST be made every time a modeling change or feature addition results in a verified improvement in the leaderboard score or CV metric.
- **Commit Messages**: Use descriptive messages, including the metric improvement

## 7. CV-Leaderboard Alignment
- **Cross-Validation**: Internal CV performance is the primary metric. Never report a Leaderboard improvement that isn't backed by a statistically significant CV improvement.
- **Goal**: Minimize overfitting to the public leaderboard.

## 8. Feature Tracing & Rationale
- **Documentation**: Every new feature added to `src/features/` must have a documented rationale (business logic or data pattern).
- **Goal**: Avoid feature inflation and maintain interpretability.

## 9. Baseline First
- **Benchmarking**: Always establish a simple baseline (e.g., Naive mean or Linear Regression) before moving to advanced algorithms (GBDTs).
- **Goal**: Measure the real value-add of complexity.

## 10. Minimalist Config & Data-Driven Signals
- **Config Discipline**: `src/config.py` MUST only contain structural configurations (paths, directories) and core model hyperparameters (e.g., LGBM `num_leaves`, `learning_rate`).
- **Data-Driven Multipliers**: All business-logic multipliers (lifts, boosts, momentum factors, window sizes) MUST be calculated dynamically from the training data within the pipeline (e.g., `calculate_market_signals`).
- **Goal**: Ensure the pipeline automatically adapts to different data regimes (2022 vs 2023) without manual hardcoding of magic numbers.

## 11. Submission & File Management Discipline
- **Fixed Output Name**: The production submission file MUST always be named `submission.csv` in the `submissions/` directory. Avoid suffixing filenames with timestamps or versions unless explicitly requested.
- **Minimalism**: Ruthlessly avoid creating unnecessary files. If a temporary script is needed for testing, use the `scratch/` directory and ensure it is cleaned up or documented.
- **Single Source of Truth**: Keep logic in the core pipeline scripts (`builder.py`, `pipeline.py`) rather than scattering it across multiple experimental scripts.

## 12. Execution Commands (Pipeline & Evaluation)
To ensure everyone is running the exact same entry points, use the following commands from the root directory (`e:/VSCODE_WORKSPACE/NewDatathon/`):
- **To Run Evaluation (Walk-Forward CV):**
  ```bash
  python src/evaluation/evaluate.py
  ```
  *(This will train on past data and evaluate on 2021-2022 to provide the Total MAE score without data leakage).*

- **To Generate Production Submission:**
  ```bash
  python -m src.training.pipeline
  ```
  *(This trains the `ForecastingPipeline` on ALL available data up to 2022 and predicts the 2023-2024 horizon, outputting to `submissions/submission.csv`).*

## 13. Score Discrepancy Awareness
- **Local vs. Leaderboard**: Be aware that Local MAE evaluation is significantly different from the Leaderboard score (Local scores are often much better/lower). 
- **VỀ BEST**: data/best_submit/best_750k.csv = bản submission tham khảo tải về từ leaderboard của team khác, không phải code của chúng ta.
- **Caution**: Do not rely solely on Local MAE for final performance expectations; always validate with Leaderboard submissions while using CV for relative improvement tracking.

## 14. Operational Data Constraints (Critical)
- **Zero Future Data**: We have ABSOLUTELY NO DATA for the 2023-2024 period (as specified in EXAM_V1.md). This includes sales, inventory, traffic, promotions, or any other table.
- **Inference Reality**: The 2023-2024 horizon is a "blind" forecast.
- **Feature Constraint**: Any feature used during inference (2023-2024) MUST be either:
    1.  A calendar/temporal feature (Date, Month, etc.)
    2.  A static historical profile derived from 2012-2022 data (e.g., Monthly median fill_rate).
    3.  A recursive value (the model's own past predictions).
- **Prohibited**: Never attempt to load or merge any data file for dates > 2022-12-31.

## 15. Ethical Best Submission Probing
- **Signal Discovery Only**: The "Best Reference" submission (e.g., `best_624k.csv`) MUST only be used to identify potential missed signals (e.g., E-commerce events like Double Days).
- **Anti-Overfitting**: Never tune hyperparameters (Damping, Boost) or hardcode multipliers solely to match the Best Reference's values. All model adjustments must be justified by historical data patterns or statistically significant CV improvements.
- **Goal**: Maintain model robustness and avoid inheriting noise or specific biases from external submissions.

