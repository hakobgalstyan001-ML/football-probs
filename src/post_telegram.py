"""Format logged Ligue 1 predictions with Market Edge and post to Telegram.

Run from the repository root:
    python src/post_telegram.py
"""
from __future__ import annotations

import os
from pathlib import Path
import urllib.parse
import urllib.request
import json

import pandas as pd

LOG_PATH = Path("predictions_log.csv")
POSTED_CACHE_PATH = Path("data/posted_matches.txt")
MATCH_PAGE_URL = "https://www.football-data.co.uk/francem.php"


def load_env(path: str = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


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


def format_match_block(row: pd.Series) -> str:
    p_h, p_d, p_a = float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])
    m_h = float(row.get("mkt_home", p_h))
    m_d = float(row.get("mkt_draw", p_d))
    m_a = float(row.get("mkt_away", p_a))

    edges = {
        f"П1 ({row['home_team']})": (p_h - m_h, p_h, m_h),
        "Х (Ничья)": (p_d - m_d, p_d, m_d),
        f"П2 ({row['away_team']})": (p_a - m_a, p_a, m_a),
    }
    best_label, (best_edge, mod_p, mkt_p) = max(edges.items(), key=lambda x: x[1][0])
    badge = "🔥 VALUE" if best_edge >= 0.03 else "📊 Модель vs Рынок"

    return (
        f"⚽ <b>{row['home_team']} — {row['away_team']}</b>\n"
        f"🕒 {row['kickoff_uk']} (UK)\n"
        f"• Модель: П1 <b>{p_h:.1%}</b> | Х <b>{p_d:.1%}</b> | П2 <b>{p_a:.1%}</b>\n"
        f"• Рынок:  П1 {m_h:.1%} | Х {m_d:.1%} | П2 {m_a:.1%}\n"
        f"{badge}: <b>{best_label}</b> ({best_edge:+.1%})\n"
        f"🔗 <a href=\"{MATCH_PAGE_URL}\">Матч-центр и статистика лиги</a>"
    )


def main() -> None:
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    if not LOG_PATH.exists():
        print("No predictions_log.csv found yet. Nothing to post.")
        return

    df = pd.read_csv(LOG_PATH)
    pending = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")].copy()
    if pending.empty:
        print("No pending fixtures to post.")
        return

    posted_ids: set[str] = set()
    if POSTED_CACHE_PATH.exists():
        posted_ids = set(POSTED_CACHE_PATH.read_text(encoding="utf-8").splitlines())

    new_rows = []
    for _, row in pending.iterrows():
        match_key = f"{row['kickoff_uk']}|{row['home_team']}|{row['away_team']}"
        if match_key not in posted_ids:
            new_rows.append((match_key, row))

    if not new_rows:
        print("All pending fixtures were already posted to Telegram.")
        return

    header = "🇫🇷 <b>Ligue 1 — Вероятности исходов</b>\n\n"
    blocks = [format_match_block(r) for _, r in new_rows]
    message = header + "\n\n".join(blocks)

    send_telegram_message(token, chat_id, message)
    POSTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTED_CACHE_PATH.open("a", encoding="utf-8") as f:
        for match_key, _ in new_rows:
            f.write(match_key + "\n")
    print(f"Posted {len(new_rows)} match(es) to Telegram channel.")


if __name__ == "__main__":
    main()