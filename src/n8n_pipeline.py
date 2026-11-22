"""Bridge between Ligue 1 Elo+LogReg pipeline and n8n workflow nodes."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from post_telegram import (
    LOG_PATH,
    POSTED_CACHE_PATH,
    format_match_caption,
    generate_match_card,
    load_env,
)

# Route all diagnostic logs to stderr so stdout remains strictly valid JSON for n8n
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_NULL_TOKENS: frozenset[str] = frozenset({"", "nan", "none", "null", "na"})


@dataclass(frozen=True)
class BridgeConfig:
    """Immutable configuration for the n8n JSON payload bridge."""

    cards_dir: Path = Path("data/cards")
    log_path: Path = Path(LOG_PATH)
    posted_cache_path: Path = Path(POSTED_CACHE_PATH)
    value_edge_threshold: float = 0.03
    prob_decimals: int = 4


@dataclass(frozen=True)
class MatchPayload:
    """Serialized match prediction item consumed by n8n workflow nodes."""

    match_key: str
    chat_id: str
    match_date: str
    kickoff_uk: str
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    m_home: float
    m_draw: float
    m_away: float
    best_outcome: str
    best_edge: float
    is_value: bool
    card_path: str
    caption: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PostedMatchesStore:
    """Thread-safe persistence layer for deduplicating published Telegram posts."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

    def load_ids(self) -> Set[str]:
        with _CACHE_LOCK:
            if not self.cache_path.exists():
                return set()
            lines = self.cache_path.read_text(encoding="utf-8").splitlines()
            return {line.strip() for line in lines if line.strip()}

    def mark_as_posted(self, match_key: str) -> bool:
        clean_key = match_key.strip()
        if not clean_key:
            return False

        with _CACHE_LOCK:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            existing: Set[str] = set()
            if self.cache_path.exists():
                existing = {
                    line.strip()
                    for line in self.cache_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }

            if clean_key in existing:
                return False

            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{clean_key}\n")
            return True


def _safe_slug(text: str) -> str:
    """Convert match identifiers into a filesystem-safe ASCII filename stem."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")


def _resolve_chat_id() -> str:
    """Load environment variables and resolve target Telegram channel/chat ID."""
    load_env()
    return (
        os.environ.get("TELEGRAM_CHANNEL_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or ""
    ).strip()


def _filter_unplayed(df: pd.DataFrame) -> pd.DataFrame:
    """Return subset of logged fixtures that do not yet have a recorded final result."""
    if df.empty:
        return df
    if "result" not in df.columns:
        return df.copy()

    normalized = df["result"].fillna("").astype(str).str.strip().str.lower()
    return df[normalized.isin(_NULL_TOKENS)].copy()


def _build_match_payload(
    row: pd.Series,
    chat_id: str,
    config: BridgeConfig,
) -> MatchPayload:
    """Render PNG match card and compute model-vs-market value edges for one fixture."""
    match_date = str(row["match_date"]).strip()
    kickoff_uk = str(row["kickoff_uk"]).strip()
    home = str(row["home"]).strip()
    away = str(row["away"]).strip()
    match_key = f"{match_date}|{kickoff_uk}|{home}|{away}"

    slug = _safe_slug(f"{match_date}_{home}_vs_{away}")
    card_path = (config.cards_dir / f"{slug}.png").resolve()
    card_path.write_bytes(generate_match_card(row))

    p_home, p_draw, p_away = float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])
    m_home, m_draw, m_away = float(row["m_home"]), float(row["m_draw"]), float(row["m_away"])

    edges: Dict[str, float] = {
        f"1 ({home})": p_home - m_home,
        "X (Draw)": p_draw - m_draw,
        f"2 ({away})": p_away - m_away,
    }
    best_outcome, best_edge = max(edges.items(), key=lambda item: item[1])

    dec = config.prob_decimals
    return MatchPayload(
        match_key=match_key,
        chat_id=chat_id,
        match_date=match_date,
        kickoff_uk=kickoff_uk,
        home=home,
        away=away,
        p_home=round(p_home, dec),
        p_draw=round(p_draw, dec),
        p_away=round(p_away, dec),
        m_home=round(m_home, dec),
        m_draw=round(m_draw, dec),
        m_away=round(m_away, dec),
        best_outcome=best_outcome,
        best_edge=round(best_edge, dec),
        is_value=best_edge >= config.value_edge_threshold,
        card_path=str(card_path),
        caption=format_match_caption(row),
    )


def prepare_payload(
    limit: int | None = None,
    config: BridgeConfig | None = None,
) -> Dict[str, Any]:
    """Generate PNG cards and JSON payloads for all unplayed, unposted fixtures."""
    cfg = config or BridgeConfig()
    chat_id = _resolve_chat_id()

    if not cfg.log_path.exists():
        logger.warning("Predictions log not found at %s", cfg.log_path)
        output: Dict[str, Any] = {"matches": []}
        print(json.dumps(output, ensure_ascii=False))
        return output

    df = pd.read_csv(cfg.log_path)
    unplayed = _filter_unplayed(df)
    if unplayed.empty:
        logger.info("No unplayed fixtures found in %s", cfg.log_path)
        output = {"matches": []}
        print(json.dumps(output, ensure_ascii=False))
        return output

    store = PostedMatchesStore(cfg.posted_cache_path)
    posted_ids = store.load_ids()
    cfg.cards_dir.mkdir(parents=True, exist_ok=True)

    payloads: List[Dict[str, Any]] = []
    for idx in unplayed.index:
        row = unplayed.loc[idx]
        match_key = (
            f"{str(row['match_date']).strip()}|"
            f"{str(row['kickoff_uk']).strip()}|"
            f"{str(row['home']).strip()}|"
            f"{str(row['away']).strip()}"
        )
        if match_key in posted_ids:
            continue

        payload = _build_match_payload(row, chat_id=chat_id, config=cfg)
        payloads.append(payload.to_dict())

        if limit is not None and len(payloads) >= limit:
            break

    logger.info("Prepared %d unposted match payload(s).", len(payloads))
    output = {"matches": payloads}
    print(json.dumps(output, ensure_ascii=False))
    return output


def mark_posted(
    match_key: str,
    config: BridgeConfig | None = None,
) -> Dict[str, str]:
    """Persist a match key in the deduplication cache after Telegram delivery."""
    cfg = config or BridgeConfig()
    store = PostedMatchesStore(cfg.posted_cache_path)
    added = store.mark_as_posted(match_key)
    logger.info("mark_posted(%r) -> newly_added=%s", match_key, added)

    output = {"marked": match_key}
    print(json.dumps(output, ensure_ascii=False))
    return output


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI bridge between Ligue 1 ML predictions and n8n nodes."
    )
    subparsers = parser.add_subparsers(dest="command")

    prep_parser = subparsers.add_parser(
        "prepare", help="Build JSON payloads and PNG cards for unposted matches."
    )
    prep_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of matches returned (useful for demo runs).",
    )

    mark_parser = subparsers.add_parser(
        "mark_posted", help="Record a match_key as published in posted_matches.txt."
    )
    mark_parser.add_argument("match_key", help="Unique match identifier string.")

    # Preserve default invocation without explicit subcommand (`python src/n8n_pipeline.py`)
    if not argv:
        return parser.parse_args(["prepare"])
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "prepare":
        prepare_payload(limit=args.limit)
    elif args.command == "mark_posted":
        mark_posted(args.match_key)
    else:
        raise SystemExit(
            "Usage: python src/n8n_pipeline.py [prepare [--limit N] | mark_posted <match_key>]"
        )


if __name__ == "__main__":
    main()