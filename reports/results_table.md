| Model | Features | Train Matches | Test Matches | Test Log Loss |
| :--- | :--- | :---: | :---: | :---: |
| Naive Baseline (Class Priors) | Historical H/D/A frequencies | 1414 | 354 | **1.0740** |
| **Baseline: Elo + Logistic Regression** | `elo_diff`, `elo_home`, `elo_away` | 1414 | 354 | **1.0054** |
| Experiment: Elo + Form + Gradient Boosting | Elo + 5-match rolling goals & shots on target | 1414 | 354 | **1.0129** |
