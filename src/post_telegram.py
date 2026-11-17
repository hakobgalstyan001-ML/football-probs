"""Send logged upcoming Ligue 1 predictions to a Telegram channel.

Run from the repository root:
    python src/post_telegram.py
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def send_telegram_message(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_post(df: pd.DataFrame) -> str:
    lines = ["⚽️ <b>Ligue 1 — Match Outcome Probabilities (Elo + LogReg)</b>\n"]
    for row in df.itertuples(index=False):
        p_h = round(float(row.p_home) * 100, 1)
        p_d = round(float(row.p_draw) * 100, 1)
        p_a = round(float(row.p_away) * 100, 1)
        lines.append(
            f"📅 <b>{row.match_date} {row.kickoff_uk} UK</b>\n"
            f"🏟 <b>{row.home} vs {row.away}</b>\n"
            f"📊 Model:  1: <b>{p_h}%</b> | X: <b>{p_d}%</b> | 2: <b>{p_a}%</b>"
        )
        if pd.notna(row.m_home):
            m_h = round(float(row.m_home) * 100, 1)
            m_d = round(float(row.m_draw) * 100, 1)
            m_a = round(float(row.m_away) * 100, 1)
            lines.append(f"🏦 Market: 1: {m_h}% | X: {m_d}% | 2: {m_a}%")
        lines.append("")
    lines.append("🔗 Data: football-data.co.uk/francem.php")
    return "\n".join(lines).strip()


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID in .env")

    log_path = Path("predictions_log.csv")
    if not log_path.is_file():
        raise FileNotFoundError("predictions_log.csv not found. Run predict_fixtures.py first.")

    df = pd.read_csv(log_path)
    pending = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")]
    if pending.empty:
        print("No pending predictions to publish.")
        return

    text = format_post(pending)
    print("Sending Telegram post:\n" + "-" * 40)
    print(text)
    print("-" * 40)

    try:
        res = send_telegram_message(token, chat_id, text)
        print("Sent successfully! Message ID:", res["result"]["message_id"])
    except urllib.error.HTTPError as e:
        print("Telegram API error:", e.code, e.read().decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()