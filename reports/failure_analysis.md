# Failure Mode & Error Analysis (Top High-Loss Test Matches)

Matches on the chronological hold-out split where the model assigned high confidence to the favorite but an upset occurred:

| Date       | HomeTeam   | AwayTeam   |   FTHG |   FTAG | FTR   |   p_H |   p_D |   p_A |   log_loss |
|:-----------|:-----------|:-----------|-------:|-------:|:------|------:|------:|------:|-----------:|
| 04/01/2026 | Marseille  | Nantes     |      0 |      2 | A     | 0.695 | 0.165 | 0.14  |     1.9674 |
| 29/10/2025 | Metz       | Lens       |      2 |      0 | H     | 0.151 | 0.237 | 0.612 |     1.891  |
| 13/02/2026 | Rennes     | Paris SG   |      3 |      1 | H     | 0.156 | 0.214 | 0.63  |     1.8555 |
| 29/10/2025 | Marseille  | Angers     |      2 |      2 | D     | 0.709 | 0.157 | 0.134 |     1.8525 |
| 02/05/2026 | Paris SG   | Lorient    |      2 |      2 | D     | 0.717 | 0.158 | 0.126 |     1.8475 |
| 03/05/2025 | Strasbourg | Paris SG   |      2 |      1 | H     | 0.159 | 0.259 | 0.583 |     1.8414 |
| 17/05/2026 | Paris FC   | Paris SG   |      2 |      1 | H     | 0.161 | 0.207 | 0.632 |     1.826  |
| 19/04/2026 | Paris SG   | Lyon       |      1 |      2 | A     | 0.684 | 0.148 | 0.168 |     1.7847 |

## Key Structural Failure Modes Identified
- **Promoted / Cold-Start Teams:** Newly promoted clubs enter Ligue 1 with limited top-flight history; even with a promoted prior (`1420` Elo), early-season strength uncertainty remains high.
- **Missing Pre-Match Lineup & Injury Context:** Tabular match statistics (`football-data.co.uk`) do not capture late injuries, suspensions, or squad rotation before European fixtures.
- **High-Variance Low-Scoring Upsets:** Because football averages ~2.7 goals per match, a single early red card or clinical counter-attack (~12–16% tail probability) regularly produces away upsets.
