"""Format logged Ligue 1 predictions with Market Edge and post to Telegram."""
from __future__ import annotations
import json, os, urllib.parse, urllib.request
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
            os.environ.setdefault(k.strip(), v.strip().strip(chr(34)).strip(chr(39)))

def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API error: {body}")

def format_match_block(r: pd.Series) -> str:
    ko, home, away = r.iloc[1], r.iloc[3], r.iloc[4]
    ph, pd_p, pa = float(r.iloc[5]), float(r.iloc[6]), float(r.iloc[7])
    mh, md_p, ma = float(r.iloc[8]), float(r.iloc[9]), float(r.iloc[10])
    edges = {f"П1 ({home})": ph - mh, "Х (Ничья)": pd_p - md_p, f"П2 ({away})": pa - ma}
    best_lbl, best_val = max(edges.items(), key=lambda x: x[1])
    badge = "🔥 VALUE" if best_val >= 0.03 else "📊 Модель vs Рынок"
    return (
        f"⚽ <b>{home} — {away}</b>\n"
        f"🕒 {ko} (UK)\n"
        f"• Модель: П1 <b>{ph:.1%}</b> | Х <b>{pd_p:.1%}</b> | П2 <b>{pa:.1%}</b>\n"
        f"• Рынок:  П1 {mh:.1%} | Х {md_p:.1%} | П2 {ma:.1%}\n"
        f"{badge}: <b>{best_lbl}</b> ({best_val:+.1%})\n"
        f"🔗 <a href=\"{MATCH_PAGE_URL}\">Матч-центр и статистика лиги</a>"
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
    unplayed = df[df.iloc[:, 11].astype(str).str.lower().isin(["false", "0", "", "nan", "none"])]
    if unplayed.empty:
        print("No pending fixtures to post.")
        return
    posted_ids = set(POSTED_CACHE_PATH.read_text(encoding="utf-8").splitlines()) if POSTED_CACHE_PATH.exists() else set()
    new_rows = [(f"{r.iloc[1]}|{r.iloc[3]}|{r.iloc[4]}", r) for _, r in unplayed.iterrows() if f"{r.iloc[1]}|{r.iloc[3]}|{r.iloc[4]}" not in posted_ids]
    if not new_rows:
        print("All pending fixtures were already posted to Telegram.")
        return
    header = "🇫🇷 <b>Ligue 1 — Вероятности исходов</b>\n\n"
    send_telegram_message(token, chat_id, header + "\n\n".join(format_match_block(r) for _, r in new_rows))
    POSTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTED_CACHE_PATH.open("a", encoding="utf-8") as f:
        for mk, _ in new_rows:
            f.write(mk + "\n")
    print(f"Posted {len(new_rows)} match(es) to Telegram channel.")

if __name__ == "__main__":
    main()
