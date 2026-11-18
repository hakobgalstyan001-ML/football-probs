"""Experiment Week 3 (v3): Tuned Elo + Bivariate Poisson Goal Model vs Baseline.

Compares:
  1. Baseline (default Elo K=20, HFA=65 + LogReg)
  2. Tuned Elo + LogReg (K and Home Field Advantage tuned strictly on train)
  3. Poisson Goal Model (predicts expected home/away goals -> 1X2 outcome matrix)
  4. Hybrid Ensemble (Tuned Elo LogReg + Poisson)

Run from the repository root:
    python src/model_v3.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

OUTCOMES = ["H", "D", "A"]
ELO_START = 1500.0
MAX_GOALS = 9


def load_matches(raw_dir: str = "data/raw/*.csv") -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(raw_dir)):
        df = pd.read_csv(path).copy()
        df["season"] = Path(path).stem.replace("ligue1_", "")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["dt"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    return data.sort_values(["dt", "HomeTeam"]).reset_index(drop=True)


def compute_elo_features(df: pd.DataFrame, k: float = 20.0, hfa: float = 65.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    elo_h, elo_a = [], []

    for row in df.itertuples(index=False):
        rh = ratings.get(row.HomeTeam, ELO_START)
        ra = ratings.get(row.AwayTeam, ELO_START)
        elo_h.append(rh)
        elo_a.append(ra)

        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + hfa)) / 400.0))
        actual_h = 1.0 if row.FTR == "H" else (0.5 if row.FTR == "D" else 0.0)
        delta = k * (actual_h - exp_h)
        ratings[row.HomeTeam] = rh + delta
        ratings[row.AwayTeam] = ra - delta

    diff = np.array(elo_h) - np.array(elo_a)
    return pd.DataFrame({"elo_diff": diff, "abs_elo_diff": np.abs(diff)}, index=df.index)


def tune_elo_hyperparams(df: pd.DataFrame, train_end_season: str) -> tuple[float, float]:
    """Tune Elo K and Home Field Advantage strictly inside training data (val = season before test)."""
    seasons = sorted(df["season"].unique())
    train_seasons = [s for s in seasons if s < train_end_season]
    val_season = train_seasons[-1]
    sub_train_mask = df["season"] < val_season
    sub_val_mask = df["season"] == val_season

    y_tr = df.loc[sub_train_mask, "FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()
    y_va = df.loc[sub_val_mask, "FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()

    best_loss = float("inf")
    best_params = (20.0, 65.0)

    for k in [10.0, 14.0, 18.0, 22.0, 26.0]:
        for hfa in [40.0, 55.0, 70.0, 85.0]:
            feats = compute_elo_features(df, k=k, hfa=hfa)
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(feats.loc[sub_train_mask], y_tr)
            probs = clf.predict_proba(feats.loc[sub_val_mask])
            loss = log_loss(y_va, probs)
            if loss < best_loss:
                best_loss = loss
                best_params = (k, hfa)

    return best_params


def fit_predict_poisson(
    df: pd.DataFrame, elo_feats: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series
) -> np.ndarray:
    """Predict expected home/away goals via regularized Poisson regression and convert to 1X2 probs."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    team_dummies = enc.fit_transform(df[["HomeTeam", "AwayTeam"]])

    X_all = np.hstack([elo_feats[["elo_diff"]].to_numpy() / 100.0, team_dummies])
    X_train, X_test = X_all[train_mask], X_all[test_mask]

    y_home_goals = df.loc[train_mask, "FTHG"].to_numpy()
    y_away_goals = df.loc[train_mask, "FTAG"].to_numpy()

    # Exponential time decay so recent matches weigh slightly more
    n_tr = X_train.shape[0]
    sample_weights = np.exp(np.linspace(-1.5, 0.0, n_tr))

    model_h = PoissonRegressor(alpha=1.5, max_iter=1000)
    model_a = PoissonRegressor(alpha=1.5, max_iter=1000)
    model_h.fit(X_train, y_home_goals, sample_weight=sample_weights)
    model_a.fit(X_train, y_away_goals, sample_weight=sample_weights)

    lam_h = np.clip(model_h.predict(X_test), 0.1, 6.0)
    lam_a = np.clip(model_a.predict(X_test), 0.1, 6.0)

    goals = np.arange(MAX_GOALS)
    probs_1x2 = np.zeros((len(lam_h), 3))
    for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
        pmf_h = poisson.pmf(goals, lh)
        pmf_a = poisson.pmf(goals, la)
        mat = np.outer(pmf_h, pmf_a)
        mat /= mat.sum()
        p_h = np.tril(mat, -1).sum()
        p_d = np.trace(mat)
        p_a = np.triu(mat, 1).sum()
        probs_1x2[i] = [p_h, p_d, p_a]

    return probs_1x2


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
    diff = ll_m - ll_b
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def market_probs(df: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / df[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def evaluate_season(df: pd.DataFrame, test_season: str) -> None:
    train_mask = df["season"] < test_season
    test_mask = df["season"] == test_season

    y_train = df.loc[train_mask, "FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()
    y_test = df.loc[test_mask, "FTR"].map({c: i for i, c in enumerate(OUTCOMES)}).to_numpy()

    # 1. Baseline (default Elo K=20, HFA=65 + LogReg)
    base_feats = compute_elo_features(df, k=20.0, hfa=65.0)
    m_base = LogisticRegression(C=1.0, max_iter=1000)
    m_base.fit(base_feats.loc[train_mask], y_train)
    p_base = m_base.predict_proba(base_feats.loc[test_mask])

    # 2. Tuned Elo + LogReg
    best_k, best_hfa = tune_elo_hyperparams(df, test_season)
    tuned_feats = compute_elo_features(df, k=best_k, hfa=best_hfa)
    m_tuned = LogisticRegression(C=1.0, max_iter=1000)
    m_tuned.fit(tuned_feats.loc[train_mask], y_train)
    p_tuned = m_tuned.predict_proba(tuned_feats.loc[test_mask])

    # 3. Poisson Goal Model
    p_pois = fit_predict_poisson(df, tuned_feats, train_mask, test_mask)

    # 4. Hybrid Ensemble (0.6 Tuned Elo + 0.4 Poisson)
    p_ens = 0.6 * p_tuned + 0.4 * p_pois

    # Market reference
    p_mkt = market_probs(df.loc[test_mask])

    print(f"\n=== Test Season: {test_season} (n={test_mask.sum()} matches) | Tuned Elo: K={best_k}, HFA={best_hfa} ===")
    rows = [
        ("Market (de-vigged)", log_loss(y_test, p_mkt), multiclass_brier(y_test, p_mkt)),
        ("Baseline (Elo K=20 + LogReg)", log_loss(y_test, p_base), multiclass_brier(y_test, p_base)),
        ("Model v3a (Tuned Elo + LogReg)", log_loss(y_test, p_tuned), multiclass_brier(y_test, p_tuned)),
        ("Model v3b (Poisson Goals)", log_loss(y_test, p_pois), multiclass_brier(y_test, p_pois)),
        ("Model v3c (Ensemble Elo+Poisson)", log_loss(y_test, p_ens), multiclass_brier(y_test, p_ens)),
    ]
    res_df = pd.DataFrame(rows, columns=["Model", "LogLoss", "Brier"])
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for name, probs in [("v3a (Tuned Elo)", p_tuned), ("v3b (Poisson)", p_pois), ("v3c (Ensemble)", p_ens)]:
        mean_d, lo, hi = bootstrap_diff_ci(y_test, probs, p_base)
        sig = "SIGNIFICANT (<0)" if hi < 0 else ("WORSE (>0)" if lo > 0 else "within noise")
        print(f"  ΔLogLoss [{name} - Baseline]: {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] -> {sig}")


def main() -> None:
    df = load_matches()
    for season in ["2025-26", "2026-27"]:
        evaluate_season(df, season)


if __name__ == "__main__":
    main()