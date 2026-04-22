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
