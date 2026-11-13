"""Predict upcoming Ligue 1 matches and log the predictions before kickoff.

Run from the repository root:
    python src/predict_fixtures.py --fixtures data/fixtures.csv

Ratings come from every completed match in data/raw. The classifier is fitted
on all completed matches after the burn-in season. Fixture times are assumed
to be UK time. Only matches that have not kicked off are written to the log,
so a logged prediction always predates its match.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baselines import LIVE_SEASON, TEST_SEASON, TRAIN_SEASONS, load_season, outcomes
from elo_logreg import INITIAL_RATING, add_elo_features, features

UK = ZoneInfo("Europe/London")
FIXTURE_COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"]
LOG_COLS = [
    "predicted_at_utc", "match_date", "kickoff_uk", "home", "away",
    "p_home", "p_draw", "p_away", "m_home", "m_draw", "m_away", "result",
]


def read_fixtures(path: Path, div: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Fixtures file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in FIXTURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    df = df.loc[df["Div"] == div, FIXTURE_COLS].dropna(
        subset=["Date", "Time", "HomeTeam", "AwayTeam"]
    )
    naive = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.assign(
        kickoff_uk=naive.dt.tz_localize(UK, ambiguous="NaT", nonexistent="NaT")
    ).dropna(subset=["kickoff_uk"])
    return df.sort_values("kickoff_uk").reset_index(drop=True)


def fit_model() -> tuple[LogisticRegression, dict[str, float]]:
    seasons = [*TRAIN_SEASONS, TEST_SEASON, LIVE_SEASON]
    data = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    data, ratings = add_elo_features(data)
    fit_df = data[data["season"] != TRAIN_SEASONS[0]]
    clf = LogisticRegression(max_iter=1000).fit(features(fit_df), outcomes(fit_df))
    return clf, ratings


def predict(
    fixtures: pd.DataFrame, clf: LogisticRegression, ratings: dict[str, float]
) -> pd.DataFrame:
    teams = {*fixtures["HomeTeam"], *fixtures["AwayTeam"]}
    unknown = sorted(t for t in teams if t not in ratings)
    if unknown:
        print(f"WARNING: no rating for {unknown}; using {INITIAL_RATING:.0f}")
    home = fixtures["HomeTeam"].map(lambda t: ratings.get(t, INITIAL_RATING))
    away = fixtures["AwayTeam"].map(lambda t: ratings.get(t, INITIAL_RATING))
    out = fixtures.copy()
    out["elo_diff"] = ((home - away) / 400.0).to_numpy(dtype=np.float64)
    out[["p_home", "p_draw", "p_away"]] = clf.predict_proba(features(out))
    inv = 1.0 / out[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=np.float64)
    out[["m_home", "m_draw", "m_away"]] = inv / inv.sum(axis=1, keepdims=True)
    return out


def append_log(preds: pd.DataFrame, log_path: Path, now: datetime) -> int:
    upcoming = preds[preds["kickoff_uk"] > pd.Timestamp(now)]
    rows = pd.DataFrame(
        {
            "predicted_at_utc": now.isoformat(timespec="seconds"),
            "match_date": upcoming["kickoff_uk"].dt.strftime("%Y-%m-%d"),
            "kickoff_uk": upcoming["kickoff_uk"].dt.strftime("%H:%M"),
            "home": upcoming["HomeTeam"],
            "away": upcoming["AwayTeam"],
            "p_home": upcoming["p_home"].round(4),
            "p_draw": upcoming["p_draw"].round(4),
            "p_away": upcoming["p_away"].round(4),
            "m_home": upcoming["m_home"].round(4),
            "m_draw": upcoming["m_draw"].round(4),
            "m_away": upcoming["m_away"].round(4),
            "result": "",
        },
        columns=LOG_COLS,
    )
    if log_path.is_file():
        existing = pd.read_csv(log_path, dtype=str)
        seen = set(zip(existing["match_date"], existing["home"], existing["away"]))
        keep = [(d, h, a) not in seen for d, h, a in zip(rows["match_date"], rows["home"], rows["away"])]
        rows = rows[keep]
    if rows.empty:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=not log_path.is_file(), index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--div", default="F1")
    parser.add_argument("--log", type=Path, default=Path("predictions_log.csv"))
    args = parser.parse_args()

    fixtures = read_fixtures(args.fixtures, args.div)
    if fixtures.empty:
        print(f"No {args.div} fixtures in {args.fixtures}")
        return
    clf, ratings = fit_model()
    preds = predict(fixtures, clf, ratings)

    now = datetime.now(timezone.utc)
    shown = preds.assign(
        kickoff=preds["kickoff_uk"].dt.strftime("%d.%m %H:%M"),
        played=np.where(preds["kickoff_uk"] > pd.Timestamp(now), "", "past"),
    )
    for col in ("p_home", "p_draw", "p_away", "m_home", "m_draw", "m_away"):
        shown[col] = (shown[col] * 100).round(1)
    print(shown[["kickoff", "HomeTeam", "AwayTeam", "p_home", "p_draw", "p_away",
                 "m_home", "m_draw", "m_away", "played"]].to_string(index=False))
    n_logged = append_log(preds, args.log, now)
    print(f"\nLogged {n_logged} new predictions to {args.log} "
          f"(matches already kicked off are never logged)")


if __name__ == "__main__":
    main()
