"""Generate match preview cards and post Ligue 1 predictions to Telegram."""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((cx - w / 2, y), text, font=font, fill=fill)


def _team_abbr(name: str) -> str:
    cleaned = "".join(ch for ch in name.upper() if ch.isalpha())
    return (cleaned[:3] or "TM").ljust(3, " ")


def generate_match_card(r: pd.Series) -> bytes:
    """Render a 1080x560 PNG match card with 1X2 probabilities and Market Edge."""
    W, H = 1080, 560
    img = Image.new("RGB", (W, H), (14, 20, 40))
    draw = ImageDraw.Draw(img)

    # Smooth vertical-diagonal dark sports gradient
    top_col = (14, 21, 46)
    bot_col = (34, 16, 56)
    for y in range(H):
        t = y / (H - 1)
        col = tuple(int(top_col[i] * (1 - t) + bot_col[i] * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=col)

    # Subtle decorative accent bars
    draw.rectangle([0, 0, W, 8], fill=(0, 212, 170))
    draw.rounded_rectangle([36, 28, W - 36, H - 28], radius=24, outline=(58, 72, 115), width=2)

    f_tag = _load_font(22, bold=True)
    f_time = _load_font(24, bold=False)
    f_abbr = _load_font(34, bold=True)
    f_team = _load_font(38, bold=True)
    f_vs = _load_font(28, bold=True)
    f_box_title = _load_font(21, bold=True)
    f_prob = _load_font(44, bold=True)
    f_mkt = _load_font(20, bold=False)
    f_footer = _load_font(22, bold=True)

    # Top header pill
    draw.rounded_rectangle([340, 46, 740, 88], radius=18, fill=(28, 42, 82))
    _draw_centered(draw, W / 2, 54, "LIGUE 1  •  1X2 PROBABILITY MODEL", f_tag, (0, 230, 180))

    ko_text = f"{r['match_date']}   •   {r['kickoff_uk']} (UK)"
    _draw_centered(draw, W / 2, 102, ko_text, f_time, (195, 208, 235))

    home, away = str(r["home"]), str(r["away"])
    ph, pd_p, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
    mh, md_p, ma = float(r["m_home"]), float(r["m_draw"]), float(r["m_away"])

    # Left team badge + name
    left_cx, right_cx, badge_cy = 270, 810, 195
    draw.ellipse([left_cx - 46, badge_cy - 46, left_cx + 46, badge_cy + 46], fill=(26, 95, 180), outline=(110, 175, 255), width=3)
    _draw_centered(draw, left_cx, badge_cy - 19, _team_abbr(home), f_abbr, (255, 255, 255))
    _draw_centered(draw, left_cx, 252, home, f_team, (255, 255, 255))

    # Center VS pill
    draw.rounded_rectangle([W / 2 - 38, 172, W / 2 + 38, 218], radius=14, fill=(45, 58, 98))
    _draw_centered(draw, W / 2, 179, "VS", f_vs, (220, 230, 250))

    # Right team badge + name
    draw.ellipse([right_cx - 46, badge_cy - 46, right_cx + 46, badge_cy + 46], fill=(175, 45, 85), outline=(255, 130, 165), width=3)
    _draw_centered(draw, right_cx, badge_cy - 19, _team_abbr(away), f_abbr, (255, 255, 255))
    _draw_centered(draw, right_cx, 252, away, f_team, (255, 255, 255))

    # 3 Outcome probability cards
    outcomes = [
        ("1  •  HOME", ph, mh, f"1 ({home})"),
        ("X  •  DRAW", pd_p, md_p, "X (Draw)"),
        ("2  •  AWAY", pa, ma, f"2 ({away})"),
    ]
    edges = [o[1] - o[2] for o in outcomes]
    best_idx = max(range(3), key=lambda i: edges[i])

    box_w, box_h = 296, 148
    gap = 36
    start_x = (W - (3 * box_w + 2 * gap)) // 2
    box_y = 318

    for i, (label, p_mod, p_mkt, _) in enumerate(outcomes):
        bx = start_x + i * (box_w + gap)
        is_best = i == best_idx
        fill_col = (24, 44, 68) if is_best else (20, 29, 58)
        border_col = (0, 230, 165) if is_best else (64, 80, 125)
        border_w = 3 if is_best else 2

        draw.rounded_rectangle([bx, box_y, bx + box_w, box_y + box_h], radius=18, fill=fill_col, outline=border_col, width=border_w)
        cx = bx + box_w / 2
        _draw_centered(draw, cx, box_y + 14, label, f_box_title, (0, 230, 165) if is_best else (175, 192, 225))
        _draw_centered(draw, cx, box_y + 48, f"{p_mod:.1%}", f_prob, (255, 255, 255))
        _draw_centered(draw, cx, box_y + 108, f"Market: {p_mkt:.1%}", f_mkt, (165, 180, 210))

    # Bottom summary strip
    best_lbl = outcomes[best_idx][3]
    best_edge = edges[best_idx]
    footer_txt = f"MODEL EDGE:  {best_lbl} ({best_edge:+.1%})    |    Elo + Multinomial LogReg"
    _draw_centered(draw, W / 2, 488, footer_txt, f_footer, (255, 215, 90) if best_edge >= 0.03 else (190, 208, 240))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def send_telegram_photo(token: str, chat_id: str, photo_bytes: bytes, caption: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    crlf = "\r\n"

    parts: list[bytes] = []
    for field_name, field_val in [
        ("chat_id", chat_id),
        ("caption", caption),
        ("parse_mode", "HTML"),
    ]:
        parts.append(
            (
                f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="{field_name}"{crlf}{crlf}'
                f"{field_val}{crlf}"
            ).encode("utf-8")
        )

    parts.append(
        (
            f"--{boundary}{crlf}"
            f'Content-Disposition: form-data; name="photo"; filename="match_card.png"{crlf}'
            f"Content-Type: image/png{crlf}{crlf}"
        ).encode("utf-8")
        + photo_bytes
        + crlf.encode("utf-8")
    )
    parts.append(f"--{boundary}--{crlf}".encode("utf-8"))

    body_bytes = b"".join(parts)
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp_body = json.loads(resp.read().decode("utf-8"))
        if not resp_body.get("ok"):
            raise RuntimeError(f"Telegram API error: {resp_body}")


def format_match_caption(r: pd.Series) -> str:
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
        f"🇫🇷 <b>Ligue 1 — Match Outcome Probabilities</b>\n\n"
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

    POSTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    for mk, r in new_rows:
        card_png = generate_match_card(r)
        caption = format_match_caption(r)
        send_telegram_photo(token, chat_id, card_png, caption)
        with POSTED_CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(mk + "\n")
        time.sleep(0.5)

    print(f"Posted {len(new_rows)} match card(s) to Telegram channel.")


if __name__ == "__main__":
    main()