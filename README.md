# Ligue 1 Match Outcome Probabilities & Automated Telegram Pipeline

Probabilistic forecasting pipeline for French Ligue 1 (`F1`, seasons `2020-21` through `2026-27`) that estimates `1 / X / 2` (Home Win / Draw / Away Win) match outcome probabilities from historical match and bookmaker odds data (`football-data.co.uk`), compares model calibration against market benchmarks, and publishes formatted match probability cards to Telegram via `n8n`.

---

## Repository Structure

```text
football-probs/
├── data/
│   ├── raw/                        # Historical Ligue 1 CSV files from football-data.co.uk
│   └── posted_matches.txt          # Deduplication cache of already published fixtures
├── reports/
│   ├── baselines_comparison.md     # Week 2 multi-tier baseline evaluation table
│   ├── baselines_comparison.png    # Visual comparison of Log Loss, RPS, and Accuracy
│   ├── results_table.md            # Week 3 model comparison + 1,000-sample paired bootstrap CIs
│   ├── calibration_curve.png       # Multi-class probability calibration (reliability) diagram
│   └── failure_analysis.md         # Qualitative analysis of top high-loss mispredictions
├── src/
│   ├── baselines.py                # 4-tier chronological baseline benchmark & chart generator
│   ├── experiments.py              # Week 3 feature experiments, bootstrap CIs & error analysis
│   ├── predict_fixtures.py         # Upcoming fixture probability & market-edge inference
│   ├── n8n_pipeline.py             # Card rendering & Telegram payload preparation
│   └── n8n_server.py               # Multi-threaded HTTP bridge consumed by the n8n workflow
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart & Reproducibility

Follow these steps to reproduce the baseline benchmark locally from a clean clone:

1. **Clone the repository and enter the project directory:**
   ```bash
   git clone <your-repo-url>
   cd football-probs
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv

   # Windows (Git Bash):
   source .venv/Scripts/activate
   # Linux / macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the 4-tier chronological baseline benchmark:**
   ```bash
   python src/baselines.py
   ```
   This evaluates all four baseline tiers on an identical chronological hold-out split and writes:
   - `reports/baselines_comparison.md`
   - `reports/baselines_comparison.png`

4. **Start the local HTTP bridge for the `n8n` $\rightarrow$ Telegram workflow (optional):**
   Configure your Telegram bot credentials in `.env` (excluded from Git via `.gitignore`) and run:
   ```bash
   python src/n8n_server.py
   ```

---

## Dataset & Column Schema

Historical match data for French Ligue 1 (`F1`, seasons `2020-21` through `2026-27`) is sourced from the public [football-data.co.uk](https://www.football-data.co.uk/francem.php) archive. To prevent **data leakage**, all columns are strictly partitioned into pre-match features (available prior to kickoff) and post-match records (used exclusively for ground-truth targets, lagged rolling features, and post-match Elo updates):

| Category | Columns | Description | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Match Identifiers** | `Div`, `Date`, `Time`, `HomeTeam`, `AwayTeam` | Division code (`F1`), match date, kickoff time (UK), home and away club names | Constructs unique `match_key` identifiers and enforces chronological train/test splits |
| **Pre-Match Market Odds** | `B365H/D/A`, `PSH/D/A`, `AvgH/D/A`, `MaxH/D/A` | Pre-match bookmaker decimal odds (Bet365, Pinnacle, market average, and market maximum) for `1 / X / 2` outcomes | De-vigged into implied market probabilities to establish the market reference benchmark and compute model `edge` |
| **Post-Match Score & Outcome** | `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR` | Full-time and half-time goals for home/away sides and final match outcome (`H`, `D`, `A`) | Ground-truth classification target (`FTR`) and sequential Elo rating updates applied strictly **after** match completion |
| **Post-Match Match Statistics** | `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR` | Total shots, shots on target, corners, fouls, yellow cards, and red cards | Aggregated strictly as lagged rolling averages over prior $N$ matches in downstream models |
| **Prediction Log (`predictions_log.csv`)** | `p_home`, `p_draw`, `p_away`, `m_home`, `m_draw`, `m_away`, `result` | Predicted model probabilities (`p_*`), de-vigged market probabilities (`m_*`), and realized outcome | Consumed by the PNG match-card renderer and automated `n8n` $\rightarrow$ Telegram publishing pipeline |

---

## Multi-Tier Baseline Strategy (`src/baselines.py`)

Rather than benchmarking against a single naive constant predictor, `src/baselines.py` evaluates a four-tier hierarchy of baselines on an identical chronological hold-out split (`1,718` training matches / `430` test matches covering `2025-04-26` to `2026-11-08`) using **Multi-class Log Loss**, **Ranked Probability Score (RPS)**, **Brier Score**, and **Accuracy**:

1. **Historical Class Prior (Sanity Check):** Empirical training-set class frequencies for `(H, D, A)` serving as a lower-bound sanity check.
2. **Poisson Goal Model (Maher, 1982):** Regularized `PoissonRegressor` estimating home/away team attack and defense parameters to derive `1 / X / 2` outcome probabilities from a $0..9 \times 0..9$ scoreline probability grid.
3. **Dynamic Elo + Multinomial Logistic Regression (Hvattum & Arntzen, 2010):** Primary trainable ML baseline with leak-free sequential Elo updates and home-advantage adjustment.
4. **De-vigged Bookmaker Market:** Multiplicatively normalized implied probabilities from pre-match bookmaker odds (`AvgH/D/A`, `PSH/D/A`, or `B365H/D/A`), serving as the upper market reference for evaluating subsequent feature and model iterations.

### Hold-Out Benchmark Results (`N = 430` Matches)

| Model | Log Loss | RPS | Brier Score | Accuracy |
| :--- | ---: | ---: | ---: | ---: |
| 1. Historical Class Prior (Sanity) | 1.0629 | 0.2313 | 0.6425 | 0.4651 |
| 2. Poisson Goal Model (Maher, 1982) | 1.0628 | 0.2313 | 0.6425 | 0.5279 |
| 3. Dynamic Elo + LogReg (Primary ML Baseline) | 0.9985 | 0.2091 | 0.5957 | 0.5140 |
| 4. Bookmaker Market (De-vigged Reference) | 0.9687 | 0.2004 | 0.5765 | 0.5419 |

**Key takeaway:** While the static Poisson goal model achieves higher top-1 accuracy (`52.79%`) than Dynamic Elo + LogReg (`51.40%`), its static multi-season team strength parameters fail to improve probability calibration (`1.0628` vs. `1.0629` Log Loss). Sequential Elo updates track team form dynamically and reduce Log Loss to `0.9985` (`0.2091` RPS), leaving a `0.0298` Log Loss gap to the de-vigged bookmaker market.

---

## Automated `n8n` $\rightarrow$ Telegram Pipeline

- **Inference & Edge Calculation:** `src/predict_fixtures.py` computes `1 / X / 2` probabilities using the calibrated baseline model and compares them against de-vigged pre-match odds.
- **Card Rendering & Local API:** `src/n8n_server.py` and `src/n8n_pipeline.py` generate match probability PNG cards and expose local HTTP endpoints for `n8n`.
- **Idempotent Delivery:** `n8n` processes fixtures sequentially (`Items per Batch = 1`) and records published match keys in `data/posted_matches.txt` to prevent duplicate posts across scheduled runs.

---

## Related Work

1. **Maher, M. J. (1982).** *Modelling association football scores.* Statistica Neerlandica, 36(3), 109–118. — Introduces independent Poisson goal-scoring distributions parameterized by team-specific attacking and defensive strengths (implemented as our Tier 2 statistical goal baseline in `src/baselines.py`).
2. **Dixon, M. J., & Coles, S. G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society: Series C, 46(2), 265–280. — Extends the bivariate Poisson framework with low-score correlation adjustments (`0–0`, `1–0`, `0–1`, `1–1`), exponential time-decay weighting, and systematic probability comparison against bookmaker odds.
3. **Hvattum, L. M., & Arntzen, H. (2010).** *Using ELO ratings for match result prediction in association football.* International Journal of Forecasting, 26(3), 460–470. — Demonstrates that sequential Elo rating differentials paired with ordinal/multinomial logistic regression yield well-calibrated `1 / X / 2` match outcome probabilities (implemented as our Tier 3 primary ML baseline).
4. **Constantinou, A. C., & Fenton, N. E. (2012).** *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models.* Journal of Quantitative Analysis in Sports, 8(1). — Establishes why strictly proper and order-sensitive scoring rules (**Multi-class Log Loss** and **Ranked Probability Score**) are required over raw classification accuracy for evaluating probabilistic football forecasts.
5. **Berrar, D., Lopes, P., & Dubitzky, W. (2019).** *Incorporating domain knowledge in machine learning for soccer outcome prediction.* Machine Learning, 108(1), 97–126. — Findings from the *Soccer Prediction Challenge* showing that domain-engineered tabular features (dynamic strength ratings, goal differentials, and rolling shots on target `HST/AST`) outperform complex deep architectures on limited match histories.