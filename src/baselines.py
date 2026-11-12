"""Baselines for Ligue 1 match outcome probabilities.

Run from the repository root:  python src/baselines.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

DATA_DIR = Path("data/raw")
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"
LIVE_SEASON = "2026-27"
OUTCOMES = {"H": 0, "D": 1, "A": 2}
ODDS_COLS = ["AvgCH", "AvgCD", "AvgCA"]
REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTR", *ODDS_COLS]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def load_season(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"ligue1_{season}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    df = df[REQUIRED_COLS].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["season"] = season
    return df.sort_values("Date").reset_index(drop=True)


def outcomes(df: pd.DataFrame) -> IntArray:
    return df["FTR"].map(OUTCOMES).to_numpy(dtype=np.int64)


def market_probs(df: pd.DataFrame) -> FloatArray:
    inv = 1.0 / df[ODDS_COLS].to_numpy(dtype=np.float64)
    return inv / inv.sum(axis=1, keepdims=True)


def frequency_probs(train: pd.DataFrame, n_rows: int) -> FloatArray:
    freq = np.bincount(outcomes(train), minlength=3) / len(train)
    return np.tile(freq, (n_rows, 1))


def log_loss(p: FloatArray, y: IntArray) -> float:
    chosen = np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def brier(p: FloatArray, y: IntArray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(p: FloatArray, y: IntArray) -> float:
    onehot = np.eye(3)[y]
    diff = np.cumsum(p, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum(diff**2, axis=1) / 2))


def report(name: str, p: FloatArray, y: IntArray) -> None:
    print(f"  {name:<12} log loss {log_loss(p, y):.4f} | "
          f"Brier {brier(p, y):.4f} | RPS {rps(p, y):.4f}")


def main() -> None:
    train = pd.concat([load_season(s) for s in TRAIN_SEASONS], ignore_index=True)
    print(f"Train: {len(train)} matches ({TRAIN_SEASONS[0]} .. {TRAIN_SEASONS[-1]})")
    for season in (TEST_SEASON, LIVE_SEASON):
        df = load_season(season)
        y = outcomes(df)
        print(f"\n{season}: {len(df)} matches")
        report("frequency", frequency_probs(train, len(df)), y)
        report("market", market_probs(df), y)


if __name__ == "__main__":
    main()
