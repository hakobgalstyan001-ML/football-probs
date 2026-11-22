"""Multi-tier baseline benchmarks for Ligue 1 match outcome probabilities (1/X/2).

Evaluates four strictly leak-free baselines on an identical chronological hold-out split:
1. Historical Class Prior (sanity-check lower bound)
2. Independent Poisson Goal Model (Maher, 1982) via regularized PoissonRegressor
3. Dynamic Elo + Multinomial Logistic Regression (Hvattum & Arntzen, 2010)
4. De-vigged Bookmaker Closing/Pre-match Odds (market upper reference)

Exports both a Markdown summary table and a comparison bar chart under reports/.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTCOME_CLASSES: Tuple[str, str, str] = ("H", "D", "A")
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(OUTCOME_CLASSES)}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for chronological baseline evaluation."""

    raw_dir: Path = Path("data/raw")
    report_path: Path = Path("reports/baselines_comparison.md")
    chart_path: Path = Path("reports/baselines_comparison.png")
    test_ratio: float = 0.20
    elo_base: float = 1500.0
    elo_k: float = 20.0
    elo_home_adv: float = 65.0
    max_goals_grid: int = 10
    poisson_alpha: float = 1.0


def load_historical_matches(raw_dir: Path) -> pd.DataFrame:
    """Load and chronologically sort completed Ligue 1 matches from football-data.co.uk CSVs."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path).copy()
        df["source_file"] = path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    data = data[data["FTR"].isin(OUTCOME_CLASSES)].copy()
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    data["Date_parsed"] = pd.to_datetime(data["Date"], dayfirst=True, format="mixed")
    time_col = data["Time"].fillna("00:00") if "Time" in data.columns else "00:00"
    data["kickoff_sort"] = time_col.astype(str)

    data = (
        data.sort_values(["Date_parsed", "kickoff_sort", "HomeTeam", "AwayTeam"])
        .drop_duplicates(subset=["Date_parsed", "HomeTeam", "AwayTeam"], keep="last")
        .reset_index(drop=True)
    )
    return data


def compute_ranked_probability_score(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    """Compute ordinal Ranked Probability Score (Constantinou & Fenton, 2012) for H > D > A."""
    n_samples, n_classes = probs.shape
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
    y_one_hot[np.arange(n_samples), y_true_idx] = 1.0

    cum_probs = np.cumsum(probs, axis=1)[:, :-1]
    cum_true = np.cumsum(y_one_hot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_probs - cum_true) ** 2, axis=1) / (n_classes - 1)))


def compute_brier_score(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    """Compute multi-class Brier score across (H, D, A)."""
    n_samples, n_classes = probs.shape
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
    y_one_hot[np.arange(n_samples), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))


def evaluate_predictions(name: str, y_true_idx: np.ndarray, probs: np.ndarray) -> Dict[str, float | str]:
    """Evaluate a probability matrix against ground-truth outcomes."""
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    pred_idx = np.argmax(clipped, axis=1)

    return {
        "Model": name,
        "Log Loss": round(float(log_loss(y_true_idx, clipped, labels=[0, 1, 2])), 4),
        "RPS": round(compute_ranked_probability_score(y_true_idx, clipped), 4),
        "Brier Score": round(compute_brier_score(y_true_idx, clipped), 4),
        "Accuracy": round(float(accuracy_score(y_true_idx, pred_idx)), 4),
    }


def predict_prior_baseline(train_df: pd.DataFrame, n_test: int) -> np.ndarray:
    """Level 1: Empirical outcome distribution on training split."""
    counts = train_df["FTR"].value_counts().reindex(OUTCOME_CLASSES, fill_value=0).to_numpy(dtype=float)
    priors = counts / counts.sum()
    return np.tile(priors, (n_test, 1))


def predict_poisson_goal_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BenchmarkConfig,
) -> np.ndarray:
    """Level 2: Maher (1982) independent Poisson attack/defense goal baseline."""
    def _build_team_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        home_part = pd.DataFrame(
            {
                "attack": df["HomeTeam"].values,
                "defense": df["AwayTeam"].values,
                "is_home": 1.0,
                "goals": df["FTHG"].astype(float).values,
            }
        )
        away_part = pd.DataFrame(
            {
                "attack": df["AwayTeam"].values,
                "defense": df["HomeTeam"].values,
                "is_home": 0.0,
                "goals": df["FTAG"].astype(float).values,
            }
        )
        stacked = pd.concat([home_part, away_part], ignore_index=True)
        return stacked[["attack", "defense", "is_home"]], stacked["goals"].to_numpy()

    X_train_raw, y_train_goals = _build_team_rows(train_df)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_raw[["attack", "defense"]])
    X_train = np.hstack([X_train_cat, X_train_raw[["is_home"]].to_numpy()])

    poisson_model = PoissonRegressor(alpha=config.poisson_alpha, max_iter=1000)
    poisson_model.fit(X_train, y_train_goals)

    home_query = pd.DataFrame(
        {"attack": test_df["HomeTeam"].values, "defense": test_df["AwayTeam"].values, "is_home": 1.0}
    )
    away_query = pd.DataFrame(
        {"attack": test_df["AwayTeam"].values, "defense": test_df["HomeTeam"].values, "is_home": 0.0}
    )

    X_home = np.hstack([encoder.transform(home_query[["attack", "defense"]]), home_query[["is_home"]].to_numpy()])
    X_away = np.hstack([encoder.transform(away_query[["attack", "defense"]]), away_query[["is_home"]].to_numpy()])

    lambda_home = poisson_model.predict(X_home)
    lambda_away = poisson_model.predict(X_away)

    goals_range = np.arange(config.max_goals_grid)
    probs = np.zeros((len(test_df), 3), dtype=float)
    tri_away_win = np.triu_indices(config.max_goals_grid, k=1)
    tri_home_win = np.tril_indices(config.max_goals_grid, k=-1)

    for idx, (lh, la) in enumerate(zip(lambda_home, lambda_away)):
        pmf_h = poisson.pmf(goals_range, mu=lh)
        pmf_a = poisson.pmf(goals_range, mu=la)
        score_matrix = np.outer(pmf_h, pmf_a)
        score_matrix /= score_matrix.sum()

        p_home = float(score_matrix[tri_home_win].sum())
        p_draw = float(np.trace(score_matrix))
        p_away = float(score_matrix[tri_away_win].sum())
        probs[idx] = [p_home, p_draw, p_away]

    return probs


def predict_elo_logreg_baseline(
    full_df: pd.DataFrame,
    split_idx: int,
    config: BenchmarkConfig,
) -> np.ndarray:
    """Level 3: Sequential Dynamic Elo + Multinomial Logistic Regression (Hvattum & Arntzen, 2010)."""
    ratings: Dict[str, float] = {}
    elo_diffs = np.zeros(len(full_df), dtype=float)
    elo_expected_home = np.zeros(len(full_df), dtype=float)

    for i, row in full_df.iterrows():
        home, away, ftr = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        r_h = ratings.get(home, config.elo_base)
        r_a = ratings.get(away, config.elo_base)

        diff = (r_h + config.elo_home_adv) - r_a
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        elo_diffs[i] = diff
        elo_expected_home[i] = exp_h

        actual_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        goal_diff = abs(float(row["FTHG"]) - float(row["FTAG"]))
        mov_multiplier = np.log1p(goal_diff) if goal_diff > 0 else 1.0

        delta = config.elo_k * mov_multiplier * (actual_h - exp_h)
        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    X_all = np.column_stack([elo_diffs, elo_expected_home])
    y_all = full_df["FTR"].map(CLASS_TO_IDX).to_numpy()

    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_all[:split_idx], y_all[:split_idx])

    raw_probs = clf.predict_proba(X_all[split_idx:])
    ordered_probs = np.zeros_like(raw_probs)
    for col_idx, cls_id in enumerate(clf.classes_):
        ordered_probs[:, int(cls_id)] = raw_probs[:, col_idx]
    return ordered_probs


def extract_devigged_market_probs(test_df: pd.DataFrame) -> np.ndarray:
    """Level 4: Multiplicative de-vigged bookmaker odds benchmark."""
    odds_triplets = [
        ("AvgH", "AvgD", "AvgA"),
        ("PSH", "PSD", "PSA"),
        ("B365H", "B365D", "B365A"),
    ]
    probs = np.full((len(test_df), 3), 1.0 / 3.0, dtype=float)

    for idx, (_, row) in enumerate(test_df.iterrows()):
        for h_col, d_col, a_col in odds_triplets:
            if {h_col, d_col, a_col}.issubset(test_df.columns):
                vals = pd.to_numeric([row[h_col], row[d_col], row[a_col]], errors="coerce")
                if np.all(np.isfinite(vals)) and np.all(vals > 1.0):
                    implied = 1.0 / vals
                    probs[idx] = implied / implied.sum()
                    break
    return probs


def save_comparison_chart(results_df: pd.DataFrame, chart_path: Path, n_test: int) -> None:
    """Render a side-by-side PNG comparison of Log Loss, RPS, and Accuracy across baseline tiers."""
    short_labels = [
        "1. Class Prior\n(Sanity)",
        "2. Poisson Goals\n(Maher, 1982)",
        "3. Dynamic Elo\n+ LogReg (ML)",
        "4. De-vigged\nBookmaker Odds",
    ]
    colors = ["#8d99ae", "#e07a5f", "#2a9d8f", "#264653"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    metrics = [
        ("Log Loss", "Multi-class Log Loss (Lower = Better)", (0.90, 1.10)),
        ("RPS", "Ranked Probability Score (Lower = Better)", (0.18, 0.25)),
        ("Accuracy", "Top-1 Accuracy (Higher = Better)", (0.40, 0.60)),
    ]

    for ax, (col, title, ylim) in zip(axes, metrics):
        vals = results_df[col].to_numpy(dtype=float)
        bars = ax.bar(short_labels, vals, color=colors, width=0.58, edgecolor="#1d3557", linewidth=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10, color="#1d3557")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + (ylim[1] - ylim[0]) * 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1d3557",
            )

    fig.suptitle(
        f"Ligue 1 Multi-Tier Baseline Benchmark (Chronological Hold-Out: N = {n_test} Matches)",
        fontsize=12,
        fontweight="bold",
        color="#1d3557",
        y=1.02,
    )
    plt.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)


def run_baselines(config: BenchmarkConfig) -> pd.DataFrame:
    """Execute full multi-tier baseline benchmark and save markdown + PNG reports."""
    df = load_historical_matches(config.raw_dir)
    split_idx = int(len(df) * (1.0 - config.test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    y_test = test_df["FTR"].map(CLASS_TO_IDX).to_numpy()

    logger.info(
        "Chronological split: %d train matches | %d test matches (%s -> %s)",
        len(train_df),
        len(test_df),
        test_df["Date_parsed"].iloc[0].strftime("%Y-%m-%d"),
        test_df["Date_parsed"].iloc[-1].strftime("%Y-%m-%d"),
    )

    results = [
        evaluate_predictions("1. Historical Class Prior (Sanity)", y_test, predict_prior_baseline(train_df, len(test_df))),
        evaluate_predictions("2. Poisson Goal Model (Maher, 1982)", y_test, predict_poisson_goal_baseline(train_df, test_df, config)),
        evaluate_predictions("3. Dynamic Elo + LogReg (Primary ML Baseline)", y_test, predict_elo_logreg_baseline(df, split_idx, config)),
        evaluate_predictions("4. Bookmaker Market (De-vigged Reference)", y_test, extract_devigged_market_probs(test_df)),
    ]
    results_df = pd.DataFrame(results)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Baseline Benchmark Comparison (Ligue 1 Hold-Out Set)",
        "",
        f"- **Evaluation split:** {len(train_df)} train / {len(test_df)} chronological test matches "
        f"({test_df['Date_parsed'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Date_parsed'].iloc[-1].strftime('%Y-%m-%d')})",
        "- **Primary metric:** Multi-class Log Loss (lower is better), supported by Ranked Probability Score (RPS) and Brier Score.",
        "",
        results_df.to_markdown(index=False),
        "",
    ]
    config.report_path.write_text("\n".join(md_lines), encoding="utf-8")
    save_comparison_chart(results_df, config.chart_path, len(test_df))

    logger.info("Saved baseline comparison report to %s", config.report_path)
    logger.info("Saved baseline comparison chart to %s", config.chart_path)
    print("\n" + results_df.to_markdown(index=False) + "\n")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-tier Ligue 1 baseline benchmarks.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/baselines_comparison.md"))
    parser.add_argument("--chart-path", type=Path, default=Path("reports/baselines_comparison.png"))
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    run_baselines(
        BenchmarkConfig(
            raw_dir=args.raw_dir,
            report_path=args.report_path,
            chart_path=args.chart_path,
            test_ratio=args.test_ratio,
        )
    )