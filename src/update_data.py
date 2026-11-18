"""Download latest Ligue 1 results (2026-27) and upcoming fixtures from football-data.co.uk.

Run from the repository root:
    python src/update_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

CURRENT_SEASON_URL = "https://www.football-data.co.uk/mmz4281/2627/F1.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

RAW_SEASON_PATH = Path("data/raw/ligue1_2026-27.csv")
FIXTURES_PATH = Path("data/fixtures.csv")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()
    dest.write_bytes(content)
    lines = max(0, len(content.splitlines()) - 1)
    print(f"Downloaded {dest} ({lines} rows)")


def main() -> None:
    download_file(CURRENT_SEASON_URL, RAW_SEASON_PATH)
    download_file(FIXTURES_URL, FIXTURES_PATH)


if __name__ == "__main__":
    main()