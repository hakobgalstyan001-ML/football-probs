# Week 3 Experiment Results & Paired Bootstrap Validation

- **Hold-out split:** 1718 train / 430 chronological test matches (2025-04-26 to 2026-11-08)
- **Bootstrap validation:** 1000 paired resamples on test set (negative $\Delta$ Log Loss = improvement over Week 2 baseline).

| Model                                     |   Log Loss |    RPS |   Brier Score |   Accuracy | Delta Log Loss vs v1 (95% CI)   | P(Beats Baseline)   |
|:------------------------------------------|-----------:|-------:|--------------:|-----------:|:--------------------------------|:--------------------|
| Baseline (v1): Dynamic Elo + LogReg       |     0.9985 | 0.2091 |        0.5957 |     0.514  | 0.0000 (Reference)              | —                   |
| model_v2: Elo + Rolling Form + LogReg     |     0.9994 | 0.2093 |        0.5964 |     0.5093 | +0.0009 [-0.0071, +0.0093]      | 39.7%               |
| model_v3: HistGradientBoosting (Elo+Form) |     1.0186 | 0.2139 |        0.6084 |     0.514  | +0.0201 [+0.0020, +0.0394]      | 1.5%                |
| model_v4: Calibrated Elo+Form+Poisson     |     0.9984 | 0.2089 |        0.5956 |     0.507  | -0.0000 [-0.0126, +0.0122]      | 47.0%               |
| Market Reference (De-vigged Odds)         |     0.9687 | 0.2004 |        0.5765 |     0.5419 | -0.0298 [-0.0464, -0.0114]      | 100.0%              |
