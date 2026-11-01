"""Load Oracle's Elixir match data.

Oracle's Elixir publishes 12 rows per game: 10 player rows and 2 team rows.
This module turns them into two tables:

* ``games``   -- one row per game (blue vs red), with the result and team stats.
* ``players`` -- one row per player per game, with role, champion and stats.
"""
from pathlib import Path

import pandas as pd

# Team-level columns kept from the raw data.
STAT_COLS = [
    "gamelength",    # seconds
    "golddiffat15",  # gold lead at 15 minutes
    "xpdiffat15",
    "csdiffat15",
    "kills",
    "deaths",
    "dragons",
    "barons",
    "towers",
    "totalgold",
    "firstblood",
    "firstdragon",
    "firstherald",
    "firsttower",
    "visionscore",
]

# Player-level columns kept from the raw data.
PLAYER_STATS = [
    "kills", "deaths", "assists",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "dpm",           # damage to champions per minute
    "damageshare",   # share of the team's damage
    "vspm",          # vision score per minute
    "earned gpm",    # gold per minute
]

ROLES = ["top", "jng", "mid", "bot", "sup"]

# Renamed organisations, mapped to their current name so history carries over.
TEAM_ALIASES = {
    "MAD Lions KOI": "Movistar KOI",
    "Shanghai Invictus Gaming Meituan": "Invictus Gaming",
}


def load_raw(paths):
    """Read and concatenate the yearly CSVs, keeping only the columns used."""
    wanted = set(["gameid", "league", "year", "split", "playoffs", "date", "game", "patch", "side",
                  "position", "playername", "teamname", "teamid", "champion", "result"]
                 + STAT_COLS + PLAYER_STATS)
    frames = [pd.read_csv(p, low_memory=False, usecols=lambda c: c in wanted) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "teamname", "result"])
    df = df[df["teamname"] != "unknown team"]
    df["teamname"] = df["teamname"].replace(TEAM_ALIASES)
    return df


def team_rows(raw):
    """Keep only the team summary rows."""
    df = raw[raw["position"] == "team"].copy()
    keep = ["gameid", "league", "year", "split", "playoffs", "date", "game", "patch",
            "side", "teamname", "teamid", "result"] + STAT_COLS
    df = df[[c for c in keep if c in df.columns]]
    for c in STAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def player_rows(raw):
    """One row per player per game (no team summary rows)."""
    df = raw[raw["position"].isin(ROLES)].copy()
    if "playername" not in df.columns:
        return pd.DataFrame()
    keep = ["gameid", "date", "side", "teamname", "position", "playername",
            "champion", "result"] + PLAYER_STATS
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["playername"])
    for c in PLAYER_STATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def games_table(teams):
    """One row per game: blue_* and red_* columns plus blue_win."""
    blue = teams[teams["side"] == "Blue"].drop_duplicates("gameid").set_index("gameid")
    red = teams[teams["side"] == "Red"].drop_duplicates("gameid").set_index("gameid")
    common = blue.index.intersection(red.index)
    blue, red = blue.loc[common], red.loc[common]

    g = pd.DataFrame({
        "gameid": common,
        "date": blue["date"].values,
        "league": blue["league"].values,
        "game_in_series": pd.to_numeric(blue.get("game", 1), errors="coerce").fillna(1).values,
        "patch": blue["patch"].values if "patch" in blue else None,
        "blue": blue["teamname"].values,
        "red": red["teamname"].values,
        "blue_win": blue["result"].astype(int).values,
    })
    for c in STAT_COLS:
        if c in blue.columns:
            g[f"blue_{c}"] = blue[c].values
            g[f"red_{c}"] = red[c].values
    g = g.sort_values(["date", "gameid"]).reset_index(drop=True)
    return g


def load_all(data_dir):
    paths = sorted(Path(data_dir).glob("*.csv"))
    paths = [p for p in paths if p.name != "worlds_results.csv"]
    if not paths:
        raise FileNotFoundError(
            f"No CSV files in {data_dir}. Download Oracle's Elixir match data "
            "(oracleselixir.com/tools/downloads) and put the CSVs there.")
    raw = load_raw(paths)
    games = games_table(team_rows(raw))
    players = player_rows(raw)
    players = players[players["gameid"].isin(set(games["gameid"]))]
    return games, players
