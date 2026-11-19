# Ligue 1 Match Outcome Probabilities

Calibrated 1X2 (Home / Draw / Away) probability model for French Ligue 1 (`football-data.co.uk`, seasons `2020-21` to `2026-27`) with automated Market Edge comparison and Telegram channel publishing.

## Quickstart (Reproducible Pipeline)

1. **Clone & install dependencies:**
   ```bash
   git clone [https://github.com/hakobgalstyan001-ML/football-probs.git](https://github.com/hakobgalstyan001-ML/football-probs.git)
   cd football-probs
   python -m venv .venv
   source .venv/bin/activate  # Windows Git Bash: source .venv/Scripts/activate
   pip install -r requirements.txt  