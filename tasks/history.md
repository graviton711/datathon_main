# Project Execution History

## Phase 3: Pure Signal Discovery & Tri-Factor Scaling (Current)

### Milestone: Achievement of 830k Revenue MAE (2026-04-22)
- **Strategy Shift**: Transitioned from manual, heuristic-based scaling (e.g., 1.09x) to a data-driven **Tri-Factor Growth Formula**: `Revenue = Traffic * CR * AOV`.
- **Implementation**: 
    - Refactored `src/training/pipeline.py` to calculate historical growth rates for each factor independently from raw data (Web Traffic, Orders, Sales).
    - Implemented **Compounding Growth**: Projected 2024 growth by compounding the annual growth factor derived from fundamental drivers.
- **Key Discovery**: 
    - **Peak Recovery Signal**: Audited late 2022 double-days (10/10, 11/11, 12/12) and found a 9% to 50% recovery in intensity compared to 2021. This justifies predicting stronger peaks in 2023-2024 without relying on future knowledge.
    - **Holiday Budget Exhaustion**: Found a negative correlation (-0.61) between March 30 and May 1st intensities, suggesting marketing budget limits within a quarter.
- **Results**: 
    - Revenue MAE: **830,988** (Best to date).
    - Scale Ratio: **0.876x** relative to 750k benchmark.

### Milestone: Decoding the Benchmark (2026-04-22)
- **Analysis**: Determined that the "Best 750k" submission (11.3M for May 1st) requires a significantly higher growth factor than the 2022 average.
- **Decision**: Refused "code chay" (manual multipliers) in favor of searching for natural, data-driven features like `dynamic_cr_trend` to reach the **Top 1 target (610k MAE)**.

---

## Strategic Targets
1. **Benchmark**: 750k MAE.
2. **Ultimate Target (Top 1)**: 610k MAE.
