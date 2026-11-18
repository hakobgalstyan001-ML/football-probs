"""Format logged Ligue 1 predictions with Market Edge and post to Telegram."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LOG_PATH = Path("predictions_log.csv")
POSTED_CACHE_PATH = Path("data/posted_matches.txt")
MATCH_PAGE_URL = "https://www.football-data.co.uk/francem.php"


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API error: {body}")


def format_match_block(r: pd.Series) -> str:
    ko = f"{r['match_date']} {r['kickoff_uk']}"
    home, away = r["home"], r["away"]
    ph, pd_p, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
    mh, md_p, ma = float(r["m_home"]), float(r["m_draw"]), float(r["m_away"])

    edges = {
        f"1 ({home})": ph - mh,
        "X (Draw)": pd_p - md_p,
        f"2 ({away})": pa - ma,
    }
    best_lbl, best_val = max(edges.items(), key=lambda x: x[1])
    badge = "🔥 VALUE" if best_val >= 0.03 else "📊 Model vs Market"

    return (
        f"⚽ <b>{home} vs {away}</b>\n"
        f"🕒 {ko} (UK)\n"
        f"• Model:  1 <b>{ph:.1%}</b> | X <b>{pd_p:.1%}</b> | 2 <b>{pa:.1%}</b>\n"
        f"• Market: 1 {mh:.1%} | X {md_p:.1%} | 2 {ma:.1%}\n"
        f"{badge}: <b>{best_lbl}</b> ({best_val:+.1%})\n"
        f"🔗 <a href=\"{MATCH_PAGE_URL}\">Match Center & League Stats</a>"
    )


def main() -> None:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHANNEL_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID in .env")

    if not LOG_PATH.exists():
        print("No predictions_log.csv found yet. Nothing to post.")
        return

    df = pd.read_csv(LOG_PATH)
    unplayed = df[df["result"].isna() | (df["result"].astype(str).str.strip().isin(["", "nan", "None"]))]
    if unplayed.empty:
        print("No pending fixtures to post.")
        return

    posted_ids = set(POSTED_CACHE_PATH.read_text(encoding="utf-8").splitlines()) if POSTED_CACHE_PATH.exists() else set()
    new_rows = []
    for _, r in unplayed.iterrows():
        mk = f"{r['match_date']}|{r['kickoff_uk']}|{r['home']}|{r['away']}"
        if mk not in posted_ids:
            new_rows.append((mk, r))

    if not new_rows:
        print("All pending fixtures were already posted to Telegram.")
        return

    header = "🇫🇷 <b>Ligue 1 — Match Outcome Probabilities</b>\n\n"
    send_telegram_message(token, chat_id, header + "\n\n".join(format_match_block(r) for _, r in new_rows))

    POSTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTED_CACHE_PATH.open("a", encoding="utf-8") as f:
        for mk, _ in new_rows:
            f.write(mk + "\n")
    print(f"Posted {len(new_rows)} match(es) to Telegram channel.")


if __name__ == "__main__":
    main()