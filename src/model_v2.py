"""Experiment Week 3: Elo + Rolling Team Form features vs Baseline (elo+logreg).

Evaluates strictly out-of-sample on seasons 2025-26 and 2026-27 using
Log Loss, Brier Score, and 95% paired bootstrap confidence intervals.

Run from the repository root:
    python src/model_v2.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = ["H", "D", "A"]
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0
ROLL_WINDOW = 5


def load_matches(raw_dir: str = "data/raw/*.csv") -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(raw_dir)):
        df = pd.read_csv(path)
        df["season"] = Path(path).stem.replace("ligue1_", "")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["dt"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data = data.sort_values(["dt", "HomeTeam"]).reset_index(drop=True)
    return data


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    elo_h_list, elo_a_list = [], []

    for row in df.itertuples(index=False):
        rh = ratings.get(row.HomeTeam, ELO_START)
        ra = ratings.get(row.AwayTeam, ELO_START)
        elo_h_list.append(rh)
        elo_a_list.append(ra)

        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
        if row.FTR == "H":
            actual_h = 1.0
        elif row.FTR == "D":
            actual_h = 0.5
        else:
            actual_h = 0.0

        delta = ELO_K * (actual_h - exp_h)
        ratings[row.HomeTeam] = rh + delta
        ratings[row.AwayTeam] = ra - delta

    out = df.copy()
    out["elo_diff"] = np.array(elo_h_list) - np.array(elo_a_list)
    out["abs_elo_diff"] = np.abs(out["elo_diff"])
    return out


def add_rolling_form_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Compute strictly pre-match rolling averages per team (no future leakage)."""
    records = []
    for idx, row in df.iterrows():
        pts_h = 3.0 if row["FTR"] == "H" else (1.0 if row["FTR"] == "D" else 0.0)
        pts_a = 3.0 if row["FTR"] == "A" else (1.0 if row["FTR"] == "D" else 0.0)
        records.append({
            "match_idx": idx,
            "dt": row["dt"],
            "team": row["HomeTeam"],
            "is_home": 1,
            "gf": row["FTHG"],
            "ga": row["FTAG"],
            "sot_f": row["HST"],
            "sot_a": row["AST"],
            "pts": pts_h,
        })
        records.append({
            "match_idx": idx,
            "dt": row["dt"],
            "team": row["AwayTeam"],
            "is_home": 0,
            "gf": row["FTAG"],
            "ga": row["FTHG"],
            "sot_f": row["AST"],
            "sot_a": row["HST"],
            "pts": pts_a,
        })

    long_df = pd.DataFrame(records).sort_values(["team", "dt", "match_idx"]).reset_index(drop=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]
    long_df["sot_d"] = long_df["sot_f"] - long_df["sot_a"]

    for col in ["gd", "sot_d", "pts"]:
        long_df[f"roll_{col}"] = (
            long_df.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .fillna(0.0)
        )

    home_stats = long_df[long_df["is_home"] == 1].set_index("match_idx")
    away_stats = long_df[long_df["is_home"] == 0].set_index("match_idx")

    out = df.copy()
    out["gd_form_diff"] = home_stats["roll_gd"] - away_stats["roll_gd"]
    out["sot_form_diff"] = home_stats["roll_sot_d"] - away_stats["roll_sot_d"]
    out["pts_form_diff"] = home_stats["roll_pts"] - away_stats["roll_pts"]
    return out


def multiclass_brier(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(len(OUTCOMES))[y_true_idx]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def bootstrap_diff_ci(
    y_true_idx: np.ndarray,
    probs_model: np.ndarray,
    probs_base: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true_idx)
    eps = 1e-15
    ll_m = -np.log(np.clip(probs_model[np.arange(n), y_true_idx], eps, 1.0))
    ll_b = -np.log(np.clip(probs_base[np.arange(n), y_true_idx], eps, 1.0))
    diff_per_match = ll_m - ll_b

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff_per_match[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diff_per_match.mean()), float(lo), float(hi)


def market_probs(df: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / df[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def evaluate_season(df: pd.DataFrame, test_season: str) -> None:
    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()

    y_train = train["FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()
    y_test = test["FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()

    base_cols = ["elo_diff", "abs_elo_diff"]
    v2_cols = ["elo_diff", "abs_elo_diff", "gd_form_diff", "sot_form_diff", "pts_form_diff"]

    # 1. Baseline: elo+logreg
    m_base = LogisticRegression(C=1.0, max_iter=1000)
    m_base.fit(train[base_cols], y_train)
    p_base = m_base.predict_proba(test[base_cols])

    # 2. Model v2a: elo + rolling form + scaled logreg
    m_v2a = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    m_v2a.fit(train[v2_cols], y_train)
    p_v2a = m_v2a.predict_proba(test[v2_cols])

    # 3. Model v2b: elo + rolling form + HistGradientBoosting
    m_v2b = HistGradientBoostingClassifier(
        max_iter=120, max_depth=3, learning_rate=0.04, l2_regularization=5.0, random_state=42
    )
    m_v2b.fit(train[v2_cols], y_train)
    p_v2b = m_v2b.predict_proba(test[v2_cols])

    # 4. Market reference
    p_mkt = market_probs(test)

    print(f"\n=== Test Season: {test_season} (n={len(test)} matches) ===")
    rows = [
        ("Market (de-vigged)", log_loss(y_test, p_mkt), multiclass_brier(y_test, p_mkt)),
        ("Baseline (elo+logreg)", log_loss(y_test, p_base), multiclass_brier(y_test, p_base)),
        ("Model v2a (elo+form+logreg)", log_loss(y_test, p_v2a), multiclass_brier(y_test, p_v2a)),
        ("Model v2b (elo+form+gb)", log_loss(y_test, p_v2b), multiclass_brier(y_test, p_v2b)),
    ]
    res_df = pd.DataFrame(rows, columns=["Model", "LogLoss", "Brier"])
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for name, probs in [("v2a (logreg)", p_v2a), ("v2b (gb)", p_v2b)]:
        mean_d, lo, hi = bootstrap_diff_ci(y_test, probs, p_base)
        sig = "SIGNIFICANT (<0)" if hi < 0 else ("WORSE (>0)" if lo > 0 else "within noise")
        print(f"  ΔLogLoss [{name} - Baseline]: {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] -> {sig}")


def main() -> None:
    df = load_matches()
    df = add_elo_features(df)
    df = add_rolling_form_features(df, window=ROLL_WINDOW)

    for season in ["2025-26", "2026-27"]:
        evaluate_season(df, season)


if __name__ == "__main__":
    main()