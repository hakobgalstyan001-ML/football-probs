"""Docker-friendly multi-threaded HTTP bridge connecting n8n triggers to the Ligue 1 pipeline."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from post_telegram import (
    LOG_PATH,
    POSTED_CACHE_PATH,
    format_match_caption,
    generate_match_card,
    load_env,
)

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "data" / "cards"
_CACHE_LOCK = threading.Lock()

try:
    UK_TZ = ZoneInfo("Europe/London")
except Exception:
    UK_TZ = timezone.utc


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")


def _is_upcoming_match(match_date: str, kickoff_uk: str, now_utc: datetime) -> bool:
    """Return True only if the scheduled kickoff in UK local time is strictly in the future."""
    try:
        dt_naive = datetime.strptime(f"{match_date.strip()} {kickoff_uk.strip()}", "%Y-%m-%d %H:%M")
        dt_utc = dt_naive.replace(tzinfo=UK_TZ).astimezone(timezone.utc)
        return dt_utc > now_utc
    except ValueError:
        return False


def run_full_pipeline() -> dict:
    load_env(str(ROOT / ".env"))
    chat_id = os.environ.get("TELEGRAM_CHANNEL_ID") or os.environ.get("TELEGRAM_CHAT_ID") or ""

    subprocess.run([sys.executable, "src/update_data.py"], cwd=ROOT, check=False)
    subprocess.run([sys.executable, "src/elo_logreg.py"], cwd=ROOT, check=False)
    subprocess.run(
        [sys.executable, "src/predict_fixtures.py", "--fixtures", "data/fixtures.csv"],
        cwd=ROOT,
        check=False,
    )

    log_file = ROOT / LOG_PATH
    cache_file = ROOT / POSTED_CACHE_PATH
    if not log_file.exists():
        return {"matches": []}

    df = pd.read_csv(log_file)
    unplayed = df[df["result"].isna() | (df["result"].astype(str).str.strip().isin(["", "nan", "None"]))]
    if unplayed.empty:
        return {"matches": []}

    with _CACHE_LOCK:
        posted_ids = set(cache_file.read_text(encoding="utf-8").splitlines()) if cache_file.exists() else set()

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)

    matches = []
    for _, r in unplayed.iterrows():
        m_date = str(r["match_date"])
        k_uk = str(r["kickoff_uk"])
        if not _is_upcoming_match(m_date, k_uk, now_utc):
            continue

        mk = f"{m_date}|{k_uk}|{r['home']}|{r['away']}"
        if mk in posted_ids:
            continue

        slug = _safe_slug(f"{m_date}_{r['home']}_vs_{r['away']}")
        card_path = CARDS_DIR / f"{slug}.png"
        card_path.write_bytes(generate_match_card(r))

        ph, pd_p, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
        mh, md_p, ma = float(r["m_home"]), float(r["m_draw"]), float(r["m_away"])
        edges = {
            f"1 ({r['home']})": ph - mh,
            "X (Draw)": pd_p - md_p,
            f"2 ({r['away']})": pa - ma,
        }
        best_outcome, best_edge = max(edges.items(), key=lambda x: x[1])

        matches.append(
            {
                "match_key": mk,
                "chat_id": chat_id,
                "match_date": m_date,
                "kickoff_uk": k_uk,
                "home": str(r["home"]),
                "away": str(r["away"]),
                "best_outcome": best_outcome,
                "best_edge": round(best_edge, 4),
                "is_value": best_edge >= 0.03,
                "card_url": f"http://host.docker.internal:8787/card?slug={slug}",
                "caption": format_match_caption(r),
            }
        )

    return {"matches": matches}


def mark_posted(match_key: str) -> dict:
    cache_file = ROOT / POSTED_CACHE_PATH
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        existing = set(cache_file.read_text(encoding="utf-8").splitlines()) if cache_file.exists() else set()
        if match_key and match_key not in existing:
            with cache_file.open("a", encoding="utf-8") as f:
                f.write(match_key + "\n")
    return {"marked": match_key}


class PipelineHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/run":
            self._send_json(200, run_full_pipeline())
        elif parsed.path == "/card":
            slug = _safe_slug((qs.get("slug") or [""])[0])
            png_file = CARDS_DIR / f"{slug}.png"
            if png_file.exists():
                data = png_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json(404, {"error": "Card not found"})
        elif parsed.path == "/mark_posted":
            mk = (qs.get("match_key") or [""])[0]
            self._send_json(200, mark_posted(mk))
        else:
            self._send_json(404, {"error": "Not found"})

    def log_message(self, format: str, *args) -> None:
        print(f"[n8n-docker-bridge] {args[0]}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8787), PipelineHandler)
    print("Ligue 1 multi-threaded Docker ML bridge ready on http://0.0.0.0:8787 ...")
    server.serve_forever()