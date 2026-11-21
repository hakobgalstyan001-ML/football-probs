## Experiment Results (Ligue 1, 2021-22 – 2026-27)

All models are evaluated strictly chronologically (no future leakage) on the exact same train/test split using **Multiclass Log Loss** (lower is better):

| Model | Features | Train Matches | Test Matches | Test Log Loss |
| :--- | :--- | :---: | :---: | :---: |
| Naive Baseline (Class Priors) | Historical H/D/A frequencies | 1414 | 354 | **1.0740** |
| **Baseline: Elo + Logistic Regression** | `elo_diff`, `elo_home`, `elo_away` | 1414 | 354 | **1.0054** |
| Experiment: Elo + Form + Gradient Boosting | Elo + 5-match rolling goals & shots on target | 1414 | 354 | **1.0129** |

**Takeaway & Feature Freeze:** Adding rolling goal/shot features and Gradient Boosting did not beat the `Elo + Logistic Regression` baseline due to high match-to-match variance in Ligue 1. We keep `Elo + Logistic Regression` as the production model and freeze new feature additions after Week 3.

## Reproducibility & Explainability (SHAP & LIME)

Run the full evaluation and generate global `SHAP` and local `LIME` error-analysis plots in `reports/`:

```bash
pip install -r requirements.txt
python src/evaluate_and_explain.py