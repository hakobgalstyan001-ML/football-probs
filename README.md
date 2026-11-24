# Ligue 1 Match Outcome Probabilities & Automated Telegram Pipeline

Probabilistic forecasting pipeline for French Ligue 1 (`F1`, seasons `2020-21` through `2026-27`) that estimates `1 / X / 2` (Home Win / Draw / Away Win) match outcome probabilities from historical match and bookmaker odds data (`football-data.co.uk`), compares model calibration against market benchmarks, and publishes formatted match probability cards to Telegram via `n8n`.

---

## Repository Structure

```text
football-probs/
├── data/
│   └── raw/                        # Historical Ligue 1 CSV files (2020-21 to 2026-27)
├── reports/
│   ├── baselines_comparison.md     # Week 2 multi-tier baseline evaluation table
│   ├── baselines_comparison.png    # Visual comparison of Log Loss, RPS, and Accuracy
│   ├── results_table.md            # Week 3 model comparison + 1,000-sample paired bootstrap CIs
│   ├── calibration_curve.png       # Multi-class probability calibration (reliability) diagram
│   ├── failure_analysis.md         # Qualitative analysis of top high-loss mispredictions
│   └── writeup.md                  # One-page project summary (choices, results, lessons learned)
├── src/
│   ├── baselines.py                # 4-tier chronological baseline benchmark & chart generator
│   ├── experiments.py              # Week 3 feature experiments, bootstrap CIs & error analysis
│   ├── update_data.py              # Automated downloader for latest Ligue 1 results & fixtures
│   ├── predict_fixtures.py         # Upcoming fixture probability & market-edge inference
│   ├── n8n_pipeline.py             # Card rendering & Telegram payload preparation
│   └── n8n_server.py               # Multi-threaded HTTP bridge consumed by the n8n workflow
├── .env.example                    # Template for Telegram bot credentials
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart & Reproducibility

Follow these steps to reproduce all benchmarks and experiment tables locally from a clean clone:

1. **Clone the repository and enter the project directory:**
   ```bash
   git clone [https://github.com/hakobgalstyan001-ML/football-probs.git](https://github.com/hakobgalstyan001-ML/football-probs.git)
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

3. **Run the Week 2 4-tier chronological baseline benchmark:**
   ```bash
   python src/baselines.py
   ```
   Evaluates all four baseline tiers on an identical chronological hold-out split and writes `reports/baselines_comparison.md` and `reports/baselines_comparison.png`.

4. **Run the Week 3 model experiments, paired bootstrap tests, and calibration plot:**
   ```bash
   python src/experiments.py
   ```
   Evaluates candidate models against the baseline and writes `reports/results_table.md`, `reports/calibration_curve.png`, and `reports/failure_analysis.md`.

5. **Start the local HTTP bridge for the `n8n` $\rightarrow$ Telegram workflow (optional):**
   Copy `.env.example` to `.env`, fill in your Telegram bot credentials (excluded from Git via `.gitignore`), and run:
   ```bash
   cp .env.example .env
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

`src/baselines.py` evaluates a four-tier hierarchy of baselines on an identical chronological hold-out split (`1,718` training matches / `430` test matches covering `2025-04-26` to `2026-11-08`) using **Multi-class Log Loss**, **Ranked Probability Score (RPS)**, **Brier Score**, and **Accuracy**:

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

## Week 3 Experiments, Calibration & Error Analysis (`src/experiments.py`)

`src/experiments.py` evaluates three candidate models against the Week 2 baseline (`0.9985` Log Loss) on the identical `430`-match chronological split with **1,000-sample paired bootstrap confidence intervals**:

- **`model_v2` (Elo + Promoted-Team Prior `1420` + 5-Match Rolling Form + LogReg):** Tests whether short-term rolling `HST/AST`, goal differentials, points, and rest days improve responsiveness over Elo alone.
- **`model_v3` (`HistGradientBoostingClassifier` on Elo + Rolling Form):** Tests non-linear tree splits on the ~1,718-match training set.
- **`model_v4` (Calibrated Blend of Elo+Form LogReg `0.794` + Time-Decayed Poisson `0.206`):** Combines sequential Elo/form probabilities with exponentially time-decayed goal expectations.

### Experiment Comparison & Paired Bootstrap Significance (`N = 430` Matches)

| Model | Log Loss | RPS | Brier Score | Accuracy | Delta Log Loss vs v1 (95% CI) | P(Beats Baseline) |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| Baseline (v1): Dynamic Elo + LogReg | 0.9985 | 0.2091 | 0.5957 | 0.5140 | 0.0000 (Reference) | — |
| model_v2: Elo + Rolling Form + LogReg | 0.9994 | 0.2093 | 0.5964 | 0.5093 | +0.0009 [-0.0071, +0.0093] | 39.7% |
| model_v3: HistGradientBoosting (Elo+Form) | 1.0186 | 0.2139 | 0.6084 | 0.5140 | +0.0201 [+0.0020, +0.0394] | 1.5% |
| **model_v4: Calibrated Elo+Form+Poisson** | **0.9984** | **0.2089** | **0.5956** | 0.5070 | **-0.0000 [-0.0126, +0.0122]** | **47.0%** |
| Market Reference (De-vigged Odds) | 0.9687 | 0.2004 | 0.5765 | 0.5419 | -0.0298 [-0.0464, -0.0114] | 100.0% |

**Statistical & qualitative findings:** `model_v4` achieves the best ML scores across all proper scoring rules (`0.9984` Log Loss, `0.2089` RPS, `0.5956` Brier Score), though its 95% bootstrap CI (`[-0.0126, +0.0122]`) spans zero (`P(Beats Baseline) = 47.0%`), showing that sequential Elo already captures most historical match-level signal. `model_v3` overfits the sample (`+0.0201 [+0.0020, +0.0394]`). Qualitative error analysis (`reports/failure_analysis.md`) identifies late-season squad rotation (`2026-05-17 Paris FC 2–1 Paris SG`, `p_A = 0.632`) and high-favorite variance upsets (`2026-01-04 Marseille 0–2 Nantes`, `p_H = 0.695`) as the main drivers of the `0.0297` gap to closing odds.

---

## Scope Boundaries

- **Feature Freeze (Post-Week 3):** No new features are added after Week 3; Week 4 focuses on Swap Test fixes, final write-up review, and Demo Day rehearsal.
- **No Paid Lineup / Injury Feeds:** Relies exclusively on public match-level data (`football-data.co.uk`) and documents the `0.0297` Log Loss gap to closing odds as the information ceiling of tabular match logs.
- **No Automated Betting Execution:** Publishes calibrated `1 / X / 2` probabilities and market-edge diagnostics rather than placing bets.

---

## Automated `n8n` $\rightarrow$ Telegram Pipeline

- **Inference & Edge Calculation:** `src/predict_fixtures.py` computes `1 / X / 2` probabilities and compares them against de-vigged pre-match odds.
- **Card Rendering & Local API:** `src/n8n_server.py` (multi-threaded `ThreadingHTTPServer`) and `src/n8n_pipeline.py` generate match probability PNG cards, filter out already-started fixtures (`kickoff_utc > now`), and expose local HTTP endpoints for `n8n`.
- **Idempotent Delivery:** `n8n` processes fixtures sequentially and records published match keys in `data/posted_matches.txt` to prevent duplicate posts across scheduled runs.

---

## Related Work

1. **Maher, M. J. (1982).** *Modelling association football scores.* Statistica Neerlandica, 36(3), 109–118. — Introduces independent Poisson goal-scoring distributions parameterized by team-specific attacking and defensive strengths (implemented as our Tier 2 statistical goal baseline in `src/baselines.py`).
2. **Dixon, M. J., & Coles, S. G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society: Series C, 46(2), 265–280. — Extends the bivariate Poisson framework with low-score correlation adjustments (`0–0`, `1–0`, `0–1`, `1–1`), exponential time-decay weighting, and systematic probability comparison against bookmaker odds.
3. **Hvattum, L. M., & Arntzen, H. (2010).** *Using ELO ratings for match result prediction in association football.* International Journal of Forecasting, 26(3), 460–470. — Demonstrates that sequential Elo rating differentials paired with ordinal/multinomial logistic regression yield well-calibrated `1 / X / 2` match outcome probabilities (implemented as our Tier 3 primary ML baseline).
4. **Constantinou, A. C., & Fenton, N. E. (2012).** *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models.* Journal of Quantitative Analysis in Sports, 8(1). — Establishes why strictly proper and order-sensitive scoring rules (**Multi-class Log Loss** and **Ranked Probability Score**) are required over raw classification accuracy for evaluating probabilistic football forecasts.
5. **Berrar, D., Lopes, P., & Dubitzky, W. (2019).** *Incorporating domain knowledge in machine learning for soccer outcome prediction.* Machine Learning, 108(1), 97–126. — Findings from the *Soccer Prediction Challenge* showing that domain-engineered tabular features (dynamic strength ratings, goal differentials, and rolling shots on target `HST/AST`) outperform complex deep architectures on limited match histories.