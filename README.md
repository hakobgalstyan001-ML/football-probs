# Ligue 1 Match Outcome Probabilities & Market Edge Pipeline

A reproducible, end-to-end machine learning pipeline that estimates calibrated 1X2 (Home / Draw / Away) outcome probabilities for French Ligue 1 (`Div = F1`), compares model predictions against bookmaker implied probabilities (`Market Edge`), and publishes automated match previews to a Telegram channel.

- **Data Source:** [football-data.co.uk](https://www.football-data.co.uk/francem.php) (seasons `2020-21` through `2026-27`)
- **Core Stack:** Python 3, `pandas`, `numpy`, `scikit-learn`, `scipy`, Telegram Bot API

---

## 1. Architecture & Leakage Prevention

```text
football-data.co.uk (CSV)
        │
        ▼
[src/update_data.py] ──► data/raw/ligue1_*.csv & data/fixtures.csv
        │
        ▼
[src/elo_logreg.py]  ──► Sequential Elo ratings + Multinomial Logistic Regression
        │
        ▼
[src/predict_fixtures.py] ──► Out-of-sample 1X2 probabilities + Market Edge ─► predictions_log.csv
        │
        ▼
[src/post_telegram.py] ──► Deduplicated HTML match cards ─► Telegram Channel  