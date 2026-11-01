"""Generate a synthetic Oracle's Elixir-style CSV for smoke tests.

Usage: python tests/make_fake_data.py OUT.csv
Values are random and carry no real-world meaning."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worlds2026 import TEAMS  # noqa: E402

rng = np.random.default_rng(1)
leagues = {"LCK": ["Gen.G", "Hanwha Life Esports", "T1", "Dplus KIA", "KT Rolster", "BNK FEARX"],
           "LPL": ["Anyone's Legend", "Bilibili Gaming", "Top Esports", "Invictus Gaming", "JD Gaming", "Weibo Gaming"],
           "LEC": ["G2 Esports", "Movistar KOI", "Karmine Corp", "Fnatic", "Team Vitality", "Team Heretics"],
           "LCS": ["Team Liquid", "LYON", "Cloud9", "FlyQuest", "Dignitas", "Shopify Rebellion"],
           "LCP": ["CTBC Flying Oyster", "Team Secret Whales", "MVK Esports", "GAM Esports", "Fukuoka SoftBank HAWKS gaming", "DetonatioN FocusMe"],
           "CBLOL": ["LOS", "FURIA", "paiN Gaming", "LOUD", "RED Canids", "Fluxo W7M"]}
base = {"LCK": 1.0, "LPL": 0.9, "LEC": 0.3, "LCS": 0.0, "LCP": 0.1, "CBLOL": -0.3}
skill = {t: base[lg] + rng.normal(0, 0.4) for lg, ts in leagues.items() for t in ts}
rows = []
dates = pd.date_range("2025-01-10", "2026-10-10", freq="D")
gid = 0
for d in dates:
    for lg, ts in leagues.items():
        if rng.random() < 0.5:
            a, b = rng.choice(ts, 2, replace=False)
            rows.append((d, lg, a, b))
    if d.month in (5, 7, 10) and rng.random() < 0.6:
        lg1, lg2 = rng.choice(list(leagues), 2, replace=False)
        rows.append((d, "WLDs" if d.month == 10 else "MSI", rng.choice(leagues[lg1]), rng.choice(leagues[lg2])))
out = []
ROLES = ["top", "jng", "mid", "bot", "sup"]
CHAMPS = [f"Champ{i}" for i in range(60)]
for d, lg, a, b in rows:
    gid += 1
    clutch = 0.8 * ((a == "T1") - (b == "T1")) if lg == "WLDs" else 0.0  # synthetic Worlds effect for one team
    p = 1 / (1 + np.exp(-(skill[a] - skill[b] + 0.08 + clutch)))
    a_win = int(rng.random() < p)
    gd = rng.normal((skill[a] - skill[b]) * 1500, 2000)
    for team, side, win, sgn in ((a, "Blue", a_win, 1), (b, "Red", 1 - a_win, -1)):
        out.append({"gameid": f"G{gid}", "league": lg, "year": d.year, "split": "", "playoffs": 0,
                    "date": d, "game": 1, "side": side, "position": "team", "teamname": team,
                    "teamid": team, "result": win, "gamelength": 1800,
                    "golddiffat15": sgn * gd, "xpdiffat15": sgn * gd * 0.8, "csdiffat15": sgn * gd / 30,
                    "kills": rng.poisson(12 + 6 * win), "deaths": rng.poisson(12 + 6 * (1 - win)),
                    "dragons": rng.poisson(1 + 2 * win), "barons": rng.poisson(0.3 + win),
                    "towers": rng.poisson(3 + 5 * win), "firstblood": int(rng.random() < 0.5),
                    "firstdragon": int(rng.random() < 0.4 + 0.2 * win), "firstherald": int(rng.random() < 0.5),
                    "firsttower": int(rng.random() < 0.35 + 0.3 * win),
                    "totalgold": 55000 + sgn * gd + 4000 * win, "visionscore": rng.normal(200, 20)})
        for r in ROLES:
            out.append({"gameid": f"G{gid}", "league": lg, "date": d, "game": 1, "side": side,
                        "position": r, "teamname": team, "playername": f"{team}-{r}",
                        "champion": rng.choice(CHAMPS), "result": win,
                        "kills": rng.poisson(3), "deaths": rng.poisson(3), "assists": rng.poisson(5),
                        "golddiffat15": sgn * gd / 5, "xpdiffat15": sgn * gd / 6, "csdiffat15": sgn * gd / 150,
                        "dpm": rng.normal(500, 100), "damageshare": 0.2, "vspm": rng.normal(1.5, 0.3),
                        "earned gpm": rng.normal(250, 30)})
pd.DataFrame(out).to_csv(sys.argv[1], index=False)
print("fake rows:", len(out))
