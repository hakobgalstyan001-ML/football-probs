"""First model: Elo ratings + multinomial logistic regression.

Ratings are always pre-match, so no feature sees the match it predicts.
Season 2020-21 is a burn-in for the ratings only; the classifier is fitted
on 2021-22 .. 2024-25 and scored on later seasons.

Run from the repository root:  python src/elo_logreg.py
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baselines import (
    LIVE_SEASON,
    TEST_SEASON,
    TRAIN_SEASONS,
    FloatArray,
    IntArray,
    frequency_probs,
    load_season,
    market_probs,
    outcomes,
    report,
)

K_FACTOR = 20.0
HOME_ADVANTAGE = 60.0
INITIAL_RATING = 1500.0
SCORES = {"H": 1.0, "D": 0.5, "A": 0.0}
N_BOOTSTRAP = 2000


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match Elo difference (home minus away, divided by 400)."""
    ratings: dict[str, float] = {}
    diffs = np.empty(len(df), dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        home = ratings.get(row.HomeTeam, INITIAL_RATING)
        away = ratings.get(row.AwayTeam, INITIAL_RATING)
        diffs[i] = (home - away) / 400.0
        expected = 1.0 / (1.0 + 10.0 ** (-(home + HOME_ADVANTAGE - away) / 400.0))
        delta = K_FACTOR * (SCORES[row.FTR] - expected)
        ratings[row.HomeTeam] = home + delta
        ratings[row.AwayTeam] = away - delta
    out = df.copy()
    out["elo_diff"] = diffs
    return out


def features(df: pd.DataFrame) -> FloatArray:
    diff = df["elo_diff"].to_numpy(dtype=np.float64)
    return np.column_stack([diff, np.abs(diff)])


def per_match_log_loss(p: FloatArray, y: IntArray) -> FloatArray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def bootstrap_diff(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOTSTRAP, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    seasons = [*TRAIN_SEASONS, TEST_SEASON, LIVE_SEASON]
    data = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    data = add_elo_features(data)

    fit_df = data[data["season"].isin(TRAIN_SEASONS[1:])]
    freq_df = data[data["season"].isin(TRAIN_SEASONS)]
    clf = LogisticRegression(max_iter=1000).fit(features(fit_df), outcomes(fit_df))
    print(f"Classifier fitted on {len(fit_df)} matches ({TRAIN_SEASONS[1]} .. {TRAIN_SEASONS[-1]})")

    for season in (TEST_SEASON, LIVE_SEASON):
        df = data[data["season"] == season]
        y = outcomes(df)
        p_model = clf.predict_proba(features(df))
        p_freq = frequency_probs(freq_df, len(df))
        p_market = market_probs(df)
        print(f"\n{season}: {len(df)} matches")
        report("frequency", p_freq, y)
        report("elo+logreg", p_model, y)
        report("market", p_market, y)
        ll_model = per_match_log_loss(p_model, y)
        for name, p in (("frequency", p_freq), ("market", p_market)):
            mean, lo, hi = bootstrap_diff(ll_model, per_match_log_loss(p, y))
            print(f"  model - {name:<9} {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
