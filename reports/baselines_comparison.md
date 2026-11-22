# Baseline Benchmark Comparison (Ligue 1 Hold-Out Set)

- **Evaluation split:** 1718 train / 430 chronological test matches (2025-04-26 to 2026-11-08)
- **Primary metric:** Multi-class Log Loss (lower is better), supported by Ranked Probability Score (RPS) and Brier Score.

| Model                                         |   Log Loss |    RPS |   Brier Score |   Accuracy |
|:----------------------------------------------|-----------:|-------:|--------------:|-----------:|
| 1. Historical Class Prior (Sanity)            |     1.0629 | 0.2313 |        0.6425 |     0.4651 |
| 2. Poisson Goal Model (Maher, 1982)           |     1.0628 | 0.2313 |        0.6425 |     0.5279 |
| 3. Dynamic Elo + LogReg (Primary ML Baseline) |     0.9985 | 0.2091 |        0.5957 |     0.514  |
| 4. Bookmaker Market (De-vigged Reference)     |     0.9687 | 0.2004 |        0.5765 |     0.5419 |
