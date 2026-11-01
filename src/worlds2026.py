"""Worlds 2026 configuration: teams, regions, Swiss results and the bracket.

``TEAMS`` maps the display name of each team to candidate names in the
Oracle's Elixir data; the first one found is used.
"""
import difflib

TEAMS = {
    "Gen.G": ["Gen.G", "Gen.G Esports"],
    "Hanwha Life Esports": ["Hanwha Life Esports"],
    "T1": ["T1"],
    "Dplus KIA": ["Dplus KIA", "Dplus Kia"],
    "Anyone's Legend": ["Anyone's Legend"],
    "Bilibili Gaming": ["Bilibili Gaming"],
    "Top Esports": ["Top Esports"],
    "Invictus Gaming": ["Invictus Gaming", "Shanghai Invictus Gaming Meituan"],
    "G2 Esports": ["G2 Esports"],
    "Movistar KOI": ["Movistar KOI", "MAD Lions KOI"],
    "Karmine Corp": ["Karmine Corp"],
    "Team Liquid": ["Team Liquid", "Team Liquid Alienware", "Team Liquid Honda"],
    "LYON": ["LYON", "Lyon"],
    "CTBC Flying Oyster": ["CTBC Flying Oyster"],
    "Team Secret Whales": ["Team Secret Whales", "Team Whales"],
    "LOS": ["LOS", "LØS", "Los Grandes"],
}

REGION = {
    "Gen.G": "LCK", "Hanwha Life Esports": "LCK", "T1": "LCK", "Dplus KIA": "LCK",
    "Anyone's Legend": "LPL", "Bilibili Gaming": "LPL", "Top Esports": "LPL", "Invictus Gaming": "LPL",
    "G2 Esports": "LEC", "Movistar KOI": "LEC", "Karmine Corp": "LEC",
    "Team Liquid": "LCS", "LYON": "LCS",
    "CTBC Flying Oyster": "LCP", "Team Secret Whales": "LCP",
    "LOS": "CBLOL",
}

# Every Swiss series: (round, team_a, team_b, winner, best_of, score as "a-b" or None if unknown)
# Sources: estnn.com results tracker and hotspawn.com recaps (retrieved 2026-11-01).
SWISS_RESULTS = [
    (1, "Gen.G", "Invictus Gaming", "Gen.G", 1, "1-0"),
    (1, "Hanwha Life Esports", "Top Esports", "Hanwha Life Esports", 1, "1-0"),
    (1, "Movistar KOI", "Bilibili Gaming", "Movistar KOI", 1, "1-0"),
    (1, "Karmine Corp", "Anyone's Legend", "Karmine Corp", 1, "1-0"),
    (1, "Dplus KIA", "G2 Esports", "Dplus KIA", 1, "1-0"),
    (1, "CTBC Flying Oyster", "Team Liquid", "CTBC Flying Oyster", 1, "1-0"),
    (1, "T1", "LOS", "T1", 1, "1-0"),
    (1, "LYON", "Team Secret Whales", "LYON", 1, "1-0"),
    (2, "Dplus KIA", "T1", "Dplus KIA", 1, "1-0"),
    (2, "Movistar KOI", "CTBC Flying Oyster", "Movistar KOI", 1, "1-0"),
    (2, "Gen.G", "Hanwha Life Esports", "Gen.G", 1, "1-0"),
    (2, "LYON", "Karmine Corp", "LYON", 1, "1-0"),
    (2, "G2 Esports", "LOS", "G2 Esports", 1, "1-0"),
    (2, "Invictus Gaming", "Team Secret Whales", "Invictus Gaming", 1, "1-0"),
    (2, "Top Esports", "Anyone's Legend", "Top Esports", 1, "1-0"),
    (2, "Bilibili Gaming", "Team Liquid", "Bilibili Gaming", 1, "1-0"),
    (3, "Dplus KIA", "Movistar KOI", "Dplus KIA", 3, "2-0"),
    (3, "Gen.G", "LYON", "Gen.G", 3, "2-1"),
    (3, "Invictus Gaming", "Hanwha Life Esports", "Invictus Gaming", 1, "1-0"),
    (3, "Top Esports", "G2 Esports", "Top Esports", 1, "1-0"),
    (3, "T1", "CTBC Flying Oyster", "T1", 1, "1-0"),
    (3, "Bilibili Gaming", "Karmine Corp", "Bilibili Gaming", 1, "1-0"),
    (3, "LOS", "Team Secret Whales", "LOS", 3, "2-0"),
    (3, "Team Liquid", "Anyone's Legend", "Team Liquid", 3, "2-0"),
    (4, "T1", "Movistar KOI", "T1", 3, "2-0"),
    (4, "Invictus Gaming", "LYON", "Invictus Gaming", 3, "2-1"),
    (4, "Karmine Corp", "CTBC Flying Oyster", "Karmine Corp", 3, "2-0"),
    (4, "Top Esports", "Bilibili Gaming", "Top Esports", 3, "2-1"),
    (4, "LOS", "Team Liquid", "LOS", 3, None),
    (4, "Hanwha Life Esports", "G2 Esports", "Hanwha Life Esports", 3, None),
    (5, "Bilibili Gaming", "Hanwha Life Esports", "Bilibili Gaming", 3, "2-0"),
    (5, "Karmine Corp", "Movistar KOI", "Karmine Corp", 3, "2-0"),
    (5, "LYON", "LOS", "LYON", 3, "2-1"),
]

# Quarterfinals (best-of-five) in bracket order: winners of QF1/QF2 meet in one
# semifinal, winners of QF3/QF4 in the other.
QUARTERFINALS = [
    ("Dplus KIA", "Bilibili Gaming"),   # DK 3-0, BLG 3-2
    ("T1", "Karmine Corp"),             # T1 3-1, KC 3-2
    ("Top Esports", "Invictus Gaming"),  # both 3-1
    ("Gen.G", "LYON"),                  # GEN 3-0, LYON 3-2
]


def swiss_records():
    """Final Swiss record per team, e.g. {"Gen.G": "3-0"}."""
    w = {t: 0 for t in TEAMS}
    l = {t: 0 for t in TEAMS}
    for _, a, b, winner, _, _ in SWISS_RESULTS:
        loser = b if winner == a else a
        w[winner] += 1
        l[loser] += 1
    return {t: f"{w[t]}-{l[t]}" for t in TEAMS}


def resolve_names(known_names, last_date):
    """Map display names to names found in the data, with warnings for misses."""
    mapping, warnings = {}, []
    for team, options in TEAMS.items():
        hit = next((o for o in options if o in known_names), None)
        if hit is None:
            close = difflib.get_close_matches(team, list(known_names), n=3, cutoff=0.6)
            if close:
                hit = max(close, key=lambda n: last_date.get(n))
                warnings.append(f"'{team}' not found exactly, using '{hit}' (closest match)")
            else:
                warnings.append(f"'{team}' not found in data, it will get a default rating")
                hit = team
        mapping[team] = hit
    return mapping, warnings
