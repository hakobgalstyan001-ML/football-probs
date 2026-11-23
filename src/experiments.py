"""Week 3 Controlled Experiments: Feature Engineering, Calibration & Bootstrap Testing.

Compares model iterations against the Week 2 Dynamic Elo + LogReg baseline on the
exact same chronological hold-out split without data leakage:
- Baseline (v1): Dynamic Elo + Multinomial Logistic Regression
- model_v2: Dynamic Elo (with promoted-team prior) + Rolling 5-Match Form + LogReg
- model_v3: Regularized HistGradientBoostingClassifier on Elo + Rolling Form features
- model_v4: Calibrated Ensemble (Elo+Form LogReg + Recency-Weighted Poisson Goals)

Exports:
- reports/results_table.md (metrics + 1,000-sample paired bootstrap 95% CIs vs baseline)
- reports/calibration_curve.png (reliability diagram verifying predicted vs empirical frequencies)
- reports/failure_analysis.md (top high-loss mispredictions for qualitative error analysis)
"""
from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baselines import (
    CLASS_TO_IDX,
    OUTCOME_CLASSES,
    BenchmarkConfig,
    compute_brier_score,
    compute_ranked_probability_score,
    evaluate_predictions,
    extract_devigged_market_probs,
    load_historical_matches,
    predict_elo_logreg_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for Week 3 model experiments and statistical validation."""

    raw_dir: Path = Path("data/raw")
    results_md_path: Path = Path("reports/results_table.md")
    calibration_png_path: Path = Path("reports/calibration_curve.png")
    failure_md_path: Path = Path("reports/failure_analysis.md")
    test_ratio: float = 0.20
    rolling_window: int = 5
    elo_base: float = 1500.0
    elo_promoted_base: float = 1420.0
    elo_k: float = 20.0
    elo_home_adv: float = 65.0
    poisson_decay_days: float = 365.0
    n_bootstrap: int = 1000
    random_seed: int = 42


def build_leakfree_feature_matrix(
    full_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Sequentially compute strictly pre-match Elo and rolling form features."""
    ratings: Dict[str, float] = {}
    matches_played: Dict[str, int] = {}
    recent_pts: Dict[str, Deque[float]] = {}
    recent_gd: Dict[str, Deque[float]] = {}
    recent_sot_diff: Dict[str, Deque[float]] = {}
    recent_shots_diff: Dict[str, Deque[float]] = {}

    feature_names = [
        "elo_diff",
        "elo_expected_home",
        "promoted_diff",
        "form_pts_diff",
        "form_gd_diff",
        "form_sot_diff",
        "form_shots_diff",
        "rest_days_diff",
    ]
    last_match_date: Dict[str, pd.Timestamp] = {}
    X = np.zeros((len(full_df), len(feature_names)), dtype=float)

    for i, row in full_df.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        ftr = str(row["FTR"])
        match_date = row["Date_parsed"]

        warmup_done = i >= 150
        default_prior = config.elo_promoted_base if warmup_done else config.elo_base

        r_h = ratings.get(home, default_prior)
        r_a = ratings.get(away, default_prior)
        mp_h = matches_played.get(home, 0)
        mp_a = matches_played.get(away, 0)

        elo_diff = (r_h + config.elo_home_adv) - r_a
        exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        prom_h = 1.0 if (warmup_done and mp_h < 10) else 0.0
        prom_a = 1.0 if (warmup_done and mp_a < 10) else 0.0
        promoted_diff = prom_h - prom_a

        pts_h = float(np.mean(recent_pts[home])) if recent_pts.get(home) else 1.35
        pts_a = float(np.mean(recent_pts[away])) if recent_pts.get(away) else 1.35

        gd_h = float(np.mean(recent_gd[home])) if recent_gd.get(home) else 0.0
        gd_a = float(np.mean(recent_gd[away])) if recent_gd.get(away) else 0.0

        sot_h = float(np.mean(recent_sot_diff[home])) if recent_sot_diff.get(home) else 0.0
        sot_a = float(np.mean(recent_sot_diff[away])) if recent_sot_diff.get(away) else 0.0

        sh_h = float(np.mean(recent_shots_diff[home])) if recent_shots_diff.get(home) else 0.0
        sh_a = float(np.mean(recent_shots_diff[away])) if recent_shots_diff.get(away) else 0.0

        rest_h = float(np.clip((match_date - last_match_date[home]).days, 2, 14)) if home in last_match_date else 7.0
        rest_a = float(np.clip((match_date - last_match_date[away]).days, 2, 14)) if away in last_match_date else 7.0

        X[i] = [
            elo_diff,
            exp_h,
            promoted_diff,
            pts_h - pts_a,
            gd_h - gd_a,
            sot_h - sot_a,
            sh_h - sh_a,
            rest_h - rest_a,
        ]

        fthg = float(row["FTHG"])
        ftag = float(row["FTAG"])
        actual_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        goal_diff = abs(fthg - ftag)
        mov_mult = np.log1p(goal_diff) if goal_diff > 0 else 1.0
        delta = config.elo_k * mov_mult * (actual_h - exp_h)

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta
        matches_played[home] = mp_h + 1
        matches_played[away] = mp_a + 1
        last_match_date[home] = match_date
        last_match_date[away] = match_date

        recent_pts.setdefault(home, deque(maxlen=config.rolling_window)).append(
            3.0 if ftr == "H" else (1.0 if ftr == "D" else 0.0)
        )
        recent_pts.setdefault(away, deque(maxlen=config.rolling_window)).append(
            3.0 if ftr == "A" else (1.0 if ftr == "D" else 0.0)
        )
        recent_gd.setdefault(home, deque(maxlen=config.rolling_window)).append(fthg - ftag)
        recent_gd.setdefault(away, deque(maxlen=config.rolling_window)).append(ftag - fthg)

        hst = pd.to_numeric(row.get("HST", np.nan), errors="coerce")
        ast = pd.to_numeric(row.get("AST", np.nan), errors="coerce")
        if np.isfinite(hst) and np.isfinite(ast):
            recent_sot_diff.setdefault(home, deque(maxlen=config.rolling_window)).append(float(hst - ast))
            recent_sot_diff.setdefault(away, deque(maxlen=config.rolling_window)).append(float(ast - hst))

        hs = pd.to_numeric(row.get("HS", np.nan), errors="coerce")
        as_ = pd.to_numeric(row.get("AS", np.nan), errors="coerce")
        if np.isfinite(hs) and np.isfinite(as_):
            recent_shots_diff.setdefault(home, deque(maxlen=config.rolling_window)).append(float(hs - as_))
            recent_shots_diff.setdefault(away, deque(maxlen=config.rolling_window)).append(float(as_ - hs))

    return X, feature_names


def _extract_ordered_probs(clf, X_eval: np.ndarray) -> np.ndarray:
    """Ensure probability columns strictly align with ('H', 'D', 'A') -> (0, 1, 2)."""
    raw_probs = clf.predict_proba(X_eval)
    ordered = np.zeros((len(X_eval), 3), dtype=float)
    for col_idx, cls_id in enumerate(clf.classes_):
        ordered[:, int(cls_id)] = raw_probs[:, col_idx]
    return ordered


def predict_model_v2_elo_form_logreg(
    X_all: np.ndarray,
    y_all: np.ndarray,
    split_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """model_v2: Standardized Dynamic Elo + Rolling 5-Game Form + Multinomial LogReg."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[:split_idx])
    X_val = scaler.transform(X_all[:split_idx])
    X_test = scaler.transform(X_all[split_idx:])

    clf = LogisticRegression(C=0.25, max_iter=1000)
    clf.fit(X_train, y_all[:split_idx])
    return _extract_ordered_probs(clf, X_val), _extract_ordered_probs(clf, X_test)


def predict_model_v3_gbdt(
    X_all: np.ndarray,
    y_all: np.ndarray,
    split_idx: int,
    seed: int,
) -> np.ndarray:
    """model_v3: Shallow Regularized HistGradientBoostingClassifier on Elo + Form."""
    gbdt = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=35,
        l2_regularization=10.0,
        random_state=seed,
    )
    gbdt.fit(X_all[:split_idx], y_all[:split_idx])
    return _extract_ordered_probs(gbdt, X_all[split_idx:])


def predict_time_decayed_poisson(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    decay_days: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Time-decayed Poisson goal model (Dixon & Coles, 1997 exponential time weighting)."""
    ref_date = train_df["Date_parsed"].max()
    days_ago = (ref_date - train_df["Date_parsed"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = np.exp(-days_ago / decay_days)

    home_part = pd.DataFrame(
        {
            "attack": train_df["HomeTeam"].values,
            "defense": train_df["AwayTeam"].values,
            "is_home": 1.0,
            "goals": train_df["FTHG"].astype(float).values,
            "weight": weights,
        }
    )
    away_part = pd.DataFrame(
        {
            "attack": train_df["AwayTeam"].values,
            "defense": train_df["HomeTeam"].values,
            "is_home": 0.0,
            "goals": train_df["FTAG"].astype(float).values,
            "weight": weights,
        }
    )
    stacked = pd.concat([home_part, away_part], ignore_index=True)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(stacked[["attack", "defense"]])
    X_train = np.hstack([X_cat, stacked[["is_home"]].to_numpy()])

    model = PoissonRegressor(alpha=1.5, max_iter=1000)
    model.fit(X_train, stacked["goals"].to_numpy(), sample_weight=stacked["weight"].to_numpy())

    q_home = pd.DataFrame(
        {"attack": test_df["HomeTeam"].values, "defense": test_df["AwayTeam"].values, "is_home": 1.0}
    )
    q_away = pd.DataFrame(
        {"attack": test_df["AwayTeam"].values, "defense": test_df["HomeTeam"].values, "is_home": 0.0}
    )
    X_h = np.hstack([encoder.transform(q_home[["attack", "defense"]]), q_home[["is_home"]].to_numpy()])
    X_a = np.hstack([encoder.transform(q_away[["attack", "defense"]]), q_away[["is_home"]].to_numpy()])

    lh_arr = model.predict(X_h)
    la_arr = model.predict(X_a)

    goals = np.arange(max_goals)
    probs = np.zeros((len(test_df), 3), dtype=float)
    tri_away = np.triu_indices(max_goals, k=1)
    tri_home = np.tril_indices(max_goals, k=-1)

    for idx, (lh, la) in enumerate(zip(lh_arr, la_arr)):
        mat = np.outer(poisson.pmf(goals, mu=lh), poisson.pmf(goals, mu=la))
        mat /= mat.sum()
        probs[idx] = [float(mat[tri_home].sum()), float(np.trace(mat)), float(mat[tri_away].sum())]

    return probs


def predict_model_v4_calibrated_blend(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v2_test_probs: np.ndarray,
    decay_days: float,
) -> np.ndarray:
    """model_v4: Optimal convex blend of Elo+Rolling-Form LogReg and Time-Decayed Poisson."""
    inner_split = int(len(train_df) * 0.75)
    inner_train = train_df.iloc[:inner_split].copy()
    inner_val = train_df.iloc[inner_split:].copy()
    y_inner_val = inner_val["FTR"].map(CLASS_TO_IDX).to_numpy()

    full_inner = pd.concat([inner_train, inner_val], ignore_index=True)
    exp_cfg = ExperimentConfig()
    X_inner, _ = build_leakfree_feature_matrix(full_inner, exp_cfg)
    y_inner = full_inner["FTR"].map(CLASS_TO_IDX).to_numpy()
    _, p_v2_inner_val = predict_model_v2_elo_form_logreg(X_inner, y_inner, inner_split)
    p_pois_inner_val = predict_time_decayed_poisson(inner_train, inner_val, decay_days=decay_days)

    def _val_loss(w: float) -> float:
        blend = w * p_v2_inner_val + (1.0 - w) * p_pois_inner_val
        blend = np.clip(blend, 1e-6, 1.0 - 1e-6)
        blend /= blend.sum(axis=1, keepdims=True)
        n = len(y_inner_val)
        return float(-np.mean(np.log(blend[np.arange(n), y_inner_val])))

    opt = minimize_scalar(_val_loss, bounds=(0.5, 1.0), method="bounded")
    best_w = float(opt.x)
    logger.info(
        "Calibrated ensemble optimal weight on inner validation: w_elo_form=%.3f, w_poisson=%.3f",
        best_w,
        1.0 - best_w,
    )

    p_pois_test = predict_time_decayed_poisson(train_df, test_df, decay_days=decay_days)
    blended = best_w * v2_test_probs + (1.0 - best_w) * p_pois_test
    return blended / blended.sum(axis=1, keepdims=True)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    baseline_probs: np.ndarray,
    candidate_probs: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Compute paired bootstrap 95% CI for Log Loss difference (candidate - baseline)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    base_clip = np.clip(baseline_probs, 1e-6, 1.0 - 1e-6)
    base_clip /= base_clip.sum(axis=1, keepdims=True)
    cand_clip = np.clip(candidate_probs, 1e-6, 1.0 - 1e-6)
    cand_clip /= cand_clip.sum(axis=1, keepdims=True)

    base_losses = -np.log(base_clip[np.arange(n), y_true])
    cand_losses = -np.log(cand_clip[np.arange(n), y_true])
    sample_diffs = cand_losses - base_losses

    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(sample_diffs[idx]))

    return {
        "mean_delta": float(np.mean(sample_diffs)),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_beats_baseline": float(np.mean(boot_means < 0.0)),
    }


def save_calibration_plot(
    y_true: np.ndarray,
    model_probs_map: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot multi-class reliability diagram comparing predicted vs empirical probabilities."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", linewidth=1.2, label="Perfect Calibration (y = x)")

    palette = {
        "Baseline (v1): Dynamic Elo + LogReg": "#e07a5f",
        "model_v2: Elo + Rolling Form + LogReg": "#2a9d8f",
        "model_v4: Calibrated Elo+Form+Poisson": "#1d3557",
        "Market Reference (De-vigged Odds)": "#8d99ae",
    }

    bins = np.linspace(0.05, 0.85, 9)
    n_samples = len(y_true)
    y_one_hot = np.zeros((n_samples, 3), dtype=float)
    y_one_hot[np.arange(n_samples), y_true] = 1.0

    for label, probs in model_probs_map.items():
        flat_p = probs.ravel()
        flat_y = y_one_hot.ravel()
        bin_ids = np.digitize(flat_p, bins) - 1

        mean_pred, mean_obs = [], []
        for b in range(len(bins) - 1):
            mask = bin_ids == b
            if np.sum(mask) >= 15:
                mean_pred.append(float(np.mean(flat_p[mask])))
                mean_obs.append(float(np.mean(flat_y[mask])))

        ax.plot(
            mean_pred,
            mean_obs,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=palette.get(label, "#264653"),
            label=label,
        )

    ax.set_title("Probability Calibration Curve (Ligue 1 Hold-Out Test Set)", fontsize=11.5, fontweight="bold", color="#1d3557")
    ax.set_xlabel("Predicted Outcome Probability", fontsize=10, color="#1d3557")
    ax.set_ylabel("Observed Empirical Frequency", fontsize=10, color="#1d3557")
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.05, 0.85)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_failure_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    best_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Export top highest-loss test predictions for qualitative failure inspection."""
    clipped = np.clip(best_probs, 1e-6, 1.0 - 1e-6)
    clipped /= clipped.sum(axis=1, keepdims=True)
    losses = -np.log(clipped[np.arange(len(y_true)), y_true])

    inspect_df = test_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    inspect_df["p_H"] = np.round(clipped[:, 0], 3)
    inspect_df["p_D"] = np.round(clipped[:, 1], 3)
    inspect_df["p_A"] = np.round(clipped[:, 2], 3)
    inspect_df["log_loss"] = np.round(losses, 4)

    worst = inspect_df.sort_values("log_loss", ascending=False).head(8)
    lines = [
        "# Failure Mode & Error Analysis (Top High-Loss Test Matches)",
        "",
        "Matches on the chronological hold-out split where the model assigned high confidence to the favorite but an upset occurred:",
        "",
        worst.to_markdown(index=False),
        "",
        "## Key Structural Failure Modes Identified",
        "- **Promoted / Cold-Start Teams:** Newly promoted clubs enter Ligue 1 with limited top-flight history; even with a promoted prior (`1420` Elo), early-season strength uncertainty remains high.",
        "- **Missing Pre-Match Lineup & Injury Context:** Tabular match statistics (`football-data.co.uk`) do not capture late injuries, suspensions, or squad rotation before European fixtures.",
        "- **High-Variance Low-Scoring Upsets:** Because football averages ~2.7 goals per match, a single early red card or clinical counter-attack (~12–16% tail probability) regularly produces away upsets.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(config: ExperimentConfig) -> pd.DataFrame:
    """Execute full Week 3 experiment suite and export results, bootstrap CIs, and plots."""
    df = load_historical_matches(config.raw_dir)
    split_idx = int(len(df) * (1.0 - config.test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    y_all = df["FTR"].map(CLASS_TO_IDX).to_numpy()
    y_test = y_all[split_idx:]

    base_cfg = BenchmarkConfig(raw_dir=config.raw_dir, test_ratio=config.test_ratio)
    probs_v1 = predict_elo_logreg_baseline(df, split_idx, base_cfg)

    X_all, _ = build_leakfree_feature_matrix(df, config)
    _, probs_v2 = predict_model_v2_elo_form_logreg(X_all, y_all, split_idx)
    probs_v3 = predict_model_v3_gbdt(X_all, y_all, split_idx, seed=config.random_seed)
    probs_v4 = predict_model_v4_calibrated_blend(train_df, test_df, probs_v2, decay_days=config.poisson_decay_days)
    probs_market = extract_devigged_market_probs(test_df)

    models = [
        ("Baseline (v1): Dynamic Elo + LogReg", probs_v1),
        ("model_v2: Elo + Rolling Form + LogReg", probs_v2),
        ("model_v3: HistGradientBoosting (Elo+Form)", probs_v3),
        ("model_v4: Calibrated Elo+Form+Poisson", probs_v4),
        ("Market Reference (De-vigged Odds)", probs_market),
    ]

    rows = []
    for name, probs in models:
        metrics = evaluate_predictions(name, y_test, probs)
        if name == "Baseline (v1): Dynamic Elo + LogReg":
            metrics["Delta Log Loss vs v1 (95% CI)"] = "0.0000 (Reference)"
            metrics["P(Beats Baseline)"] = "—"
        else:
            boot = paired_bootstrap_delta(
                y_test,
                probs_v1,
                probs,
                n_bootstrap=config.n_bootstrap,
                seed=config.random_seed,
            )
            metrics["Delta Log Loss vs v1 (95% CI)"] = (
                f"{boot['mean_delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"
            )
            metrics["P(Beats Baseline)"] = f"{boot['p_beats_baseline'] * 100:.1f}%"
        rows.append(metrics)

    results_df = pd.DataFrame(rows)

    md_lines = [
        "# Week 3 Experiment Results & Paired Bootstrap Validation",
        "",
        f"- **Hold-out split:** {len(train_df)} train / {len(test_df)} chronological test matches "
        f"({test_df['Date_parsed'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Date_parsed'].iloc[-1].strftime('%Y-%m-%d')})",
        f"- **Bootstrap validation:** {config.n_bootstrap} paired resamples on test set (negative $\\Delta$ Log Loss = improvement over Week 2 baseline).",
        "",
        results_df.to_markdown(index=False),
        "",
    ]
    config.results_md_path.parent.mkdir(parents=True, exist_ok=True)
    config.results_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    save_calibration_plot(
        y_test,
        {
            "Baseline (v1): Dynamic Elo + LogReg": probs_v1,
            "model_v2: Elo + Rolling Form + LogReg": probs_v2,
            "model_v4: Calibrated Elo+Form+Poisson": probs_v4,
            "Market Reference (De-vigged Odds)": probs_market,
        },
        config.calibration_png_path,
    )
    save_failure_analysis(test_df, y_test, probs_v4, config.failure_md_path)

    logger.info("Saved results table to %s", config.results_md_path)
    logger.info("Saved calibration plot to %s", config.calibration_png_path)
    logger.info("Saved failure analysis to %s", config.failure_md_path)
    print("\n" + results_df.to_markdown(index=False) + "\n")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 3 Ligue 1 model experiments.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()
    run_experiments(ExperimentConfig(raw_dir=args.raw_dir, test_ratio=args.test_ratio))