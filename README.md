# Ligue 1 Match Outcome Probabilities

Calibrated 1X2 (Home / Draw / Away) probability model for French Ligue 1 (`football-data.co.uk`, seasons `2020-21` to `2026-27`) with automated Market Edge comparison and Telegram channel publishing.

## Quickstart (Reproducible Pipeline)

1. **Clone & install dependencies:**
   ```bash
   git clone https://github.com/hakobgalstyan001-ML/football-probs.git
   cd football-probs
   python -m venv .venv
   source .venv/bin/activate  # Windows Git Bash: source .venv/Scripts/activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables (optional, only for Telegram posting):**
   ```bash
   cp .env.example .env
   # Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID to .env
   ```

3. **Run data update, baseline evaluation, and fixture prediction:**
   ```bash
   python src/update_data.py
   python src/elo_logreg.py
   python src/predict_fixtures.py --fixtures data/fixtures.csv
   ```h(Ğ¸€¨©A½ÍĞÁ•¹‘¥¹œÁÉ•‘¥Ñ¥½¹ÌÑ¼Q•±•É…´è¨¨(€€‰…Í (€€ÁåÑ¡½¸ÍÉŒ½Á½ÍÑ}Ñ•±•É…´¹Áä(€€€((ŒŒI•ÍÕ±ÑÌQ…‰±”€¡5•ÑÉ¥Œè5Õ±Ñ¤µ±…ÍÌ1½œ1½ÍÌ¤()±°µ½‘•±Ì…É”•Ù…±Õ…Ñ•ÍÑÉ¥Ñ±ä½ÕĞµ½˜µÍ…µÁ±”ÕÍ¥¹œ•áÁ…¹‘¥¹œµİ¥¹‘½ÜÑÉ…¥¹¥¹œİ¥Ñ é•É¼™ÕÑÕÉ”±•…­…”¸()ğ5½‘•°ğ•ÍÉ¥ÁÑ¥½¸ğ1½œ1½ÍÌ€ ÈÀÈÔ´ÈØ¤ğ1½œ1½ÍÌ€ ÈÀÈØ´ÈÜeQ¤ğ9½Ñ•Ìğ)ğ€è´´´ğ€è´´´ğ€è´´´èğ€è´´´èğ€è´´´ğ)ğ€¨©	…Í•±¥¹”€¡±¼€¬1½I•€¤¨¨ğå¹…µ¥Œ±¼É…Ñ¥¹Ì€¬5Õ±Ñ¥¹½µ¥…°1½¥ÍÑ¥ŒI•É•ÍÍ¥½¸ğ€¨¨Ä¸ÀÀĞÔ¨¨ğ€¨¨Ä¸ÀÔÄÀ€¼€À¸äÜÈÀ¨¨ğ	•ÍĞ…±¥‰É…Ñ¥½¸ì±½İ•ÍĞÙ…É¥…¹”ğ)ğ€¨©áÁ•É¥µ•¹ĞØÈ€¡µ½‘•±}ØÈ¹Áå€¤¨¨ğ±¼€¬€Ôµµ…Ñ É½±±¥¹œ½…±Ì½Í¡½ÑÌ™½É´ğ€Ä¸ÀÄÄÈğ€À¸äàÀĞğ=Ù•É™¥ÑÌÍ¡½ÉĞµÑ•É´¹½¥Í”ğ)ğ€¨©áÁ•É¥µ•¹ĞØÌ€¡µ½‘•±}ØÌ¹Áå€¤¨¨ğÉ…‘¥•¹Ğ	½½ÍÑ¥¹œ½¸±¼€¬™½É´ğ€Ä¸ÀÄàäğ€À¸äàäÄğ]½ÉÍ”…±¥‰É…Ñ¥½¸Ñ¡…¸1½I•œğ((ŒŒ…¥±ÕÉ”…Í”¹…±åÍ¥Ì€¡]¡•É”Ñ¡”5½‘•°…¥±Ì¤((´€¨©…É±äµÍ•…Í½¸€˜¹•İ±äÁÉ½µ½Ñ•Ñ•…µÌè¨¨9•İ±äÁÉ½µ½Ñ•±Õ‰ÌÍÑ…ÉĞİ¥Ñ „‘•™…Õ±ĞÁÉ¥½È±¼É…Ñ¥¹œ…¹¹¼É••¹ĞÑ½Àµ™±¥¡Ğ¡¥ÍÑ½Éä¸(´€¨©MÅÕ…É½Ñ…Ñ¥½¸€˜¥¹©ÕÉ¥•Ì‰•™½É”ÕÉ½Á•…¸™¥áÑÕÉ•Ìè¨¨MÕ‘‘•¸±¥¹•ÕÀÉ½Ñ…Ñ¥½¹Ì…É”¹½Ğ…ÁÑÕÉ•Õ¹Ñ¥°±½Í¥¹œ½‘‘Ìµ½Ù”¸((ŒŒM½Á”€˜9½¸µ½…±Ì€¡]¡…Ğ]”É”9½Ğ½¥¹œ¤((´]”…É”€¨©¹½Ğ¨¨‰Õ¥±‘¥¹œ…¸…ÕÑ½µ…Ñ•‰•ÑÑ¥¹œ‰½Ğ½È±…¥µ¥¹œÑ¼‰•…Ğ‰½½­µ…­•Èµ…É¥¹Ì±½¹œµÑ•É´¸(´Q¡”½…°¥Ì€¨©…±¥‰É…Ñ•ÁÉ½‰…‰¥±¥Ñä•ÍÑ¥µ…Ñ¥½¸¨¨…¹ÑÉ…¹ÍÁ…É•¹Ğ€¨©5…É­•Ğ‘”¨¨¡¥¡±¥¡Ñ¥¹œ€¡5½‘•°ÙÌ5…É­•Ñ€¤¸(