"""Point-in-time team and player features.

Games are replayed in chronological order. Before each game, a snapshot of
both teams is recorded; only then is the tracker updated with the result.
Every feature therefore uses information available before the game.

Team features
    elo          Elo rating with a margin-of-victory multiplier (gold gap),
                 extra weight for international events, extra weight for the
                 current Worlds, and partial regression to the league mean at
                 the start of each season.
    form_*       Exponentially weighted averages of per-game stats (win, gold /
                 XP / CS difference at 15, kill / tower / dragon / baron /
                 vision differences, first tower, first dragon).
    intl_exp     International games played in the last 365 days.
    worlds_wr    Recency-weighted Worlds win rate (half-life one year),
                 shrunk toward 50% with a prior of WORLDS_PRIOR games.

Player features (from the five players in the lineup)
    roster_elo       Mean of per-player Elo ratings, so roster changes are
                     reflected immediately.
    roster_*         Mean of per-player form (KDA, gold / XP / CS diff at 15,
                     damage per minute, damage share, vision, gold per minute).
    role_gd15_*      Gold difference at 15 for each role.
    roster_worlds_*  Recency-weighted Worlds record of the current players.
    pool_*           Distinct champions per player over the last POOL_DAYS
                     days (mean and minimum), relevant under Fearless draft.

Model inputs are blue-minus-red differences of these snapshots, plus the game
number within the series and its interaction with champion-pool depth.
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

INTERNATIONAL = {"WLDs", "MSI", "FST", "EWC", "Asia Master", "IN"}
ELO_START = 1500.0
ELO_K = 24.0
INTL_K_MULT = 1.5
# Games from the current Worlds move ratings more (on top of INTL_K_MULT).
CURRENT_WORLDS_START = pd.Timestamp("2026-10-15")
CURRENT_WORLDS_K_MULT = 2.0
BLUE_ADV = 15.0       # blue-side advantage, in Elo points
SEASON_REGRESS = 0.2  # share of the gap to the league mean removed at each new season
FORM_ALPHA = 0.15     # EWMA weight of the most recent game
WORLDS = {"WLDs"}
WORLDS_HALF_LIFE = 365  # days
WORLDS_PRIOR = 6        # pseudo-games at 50% used to shrink the Worlds win rate
POOL_DAYS = 120       # look-back window for champion pool size
ROLES = ["top", "jng", "mid", "bot", "sup"]

FORM_STATS = ["win", "gd15", "xpd15", "csd15", "kill_diff", "tower_diff",
              "dragon_diff", "baron_diff", "first_tower", "first_dragon", "vision_diff"]
PLAYER_FORM = ["kda", "gd15", "xpd15", "csd15", "dpm", "damageshare", "vspm", "gpm"]


def is_current_worlds(league, date):
    return league in {"WLDs"} and date >= CURRENT_WORLDS_START


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _ewma(store, key, val):
    if _nan(val):
        return
    store[key] = val if key not in store else (1 - FORM_ALPHA) * store[key] + FORM_ALPHA * val


def _team_stats(row, side):
    """Per-game stats for one side ("blue" or "red") of a game row."""
    other = "red" if side == "blue" else "blue"

    def d(col):
        a, b = row.get(f"{side}_{col}"), row.get(f"{other}_{col}")
        return np.nan if a is None or b is None else a - b

    return {
        "win": row["blue_win"] if side == "blue" else 1 - row["blue_win"],
        "gd15": row.get(f"{side}_golddiffat15", np.nan),
        "xpd15": row.get(f"{side}_xpdiffat15", np.nan),
        "csd15": row.get(f"{side}_csdiffat15", np.nan),
        "kill_diff": d("kills"),
        "tower_diff": d("towers"),
        "dragon_diff": d("dragons"),
        "baron_diff": d("barons"),
        "vision_diff": d("visionscore"),
        "first_tower": row.get(f"{side}_firsttower", np.nan),
        "first_dragon": row.get(f"{side}_firstdragon", np.nan),
    }


def _player_stats(p):
    k, dth, a = p.get("kills"), p.get("deaths"), p.get("assists")
    kda = np.nan if _nan(k) or _nan(dth) or _nan(a) else (k + a) / max(dth, 1)
    return {"kda": kda, "gd15": p.get("golddiffat15"), "xpd15": p.get("xpdiffat15"),
            "csd15": p.get("csdiffat15"), "dpm": p.get("dpm"), "damageshare": p.get("damageshare"),
            "vspm": p.get("vspm"), "gpm": p.get("earned gpm")}


def _mov_mult(row):
    """Elo multiplier from the final gold difference (larger wins move ratings more)."""
    gb, gr = row.get("blue_totalgold"), row.get("red_totalgold")
    if _nan(gb) or _nan(gr):
        return 1.0
    return float(np.clip(math.log1p(abs(gb - gr) / 1000) / math.log1p(5), 0.6, 1.6))


class Tracker:
    def __init__(self):
        # teams
        self.elo, self.form, self.n = {}, defaultdict(dict), defaultdict(int)
        self.league_of, self.last_date, self.intl_dates = {}, {}, defaultdict(deque)
        self.roster = {}  # team -> {role: player}, from the team's latest game
        # players
        self.pelo, self.pform, self.pn = {}, defaultdict(dict), defaultdict(int)
        self.champs = defaultdict(deque)  # player -> deque of (date, champion)
        self.player_team = {}
        self.worlds_team = defaultdict(list)    # team -> [(date, win)] at Worlds
        self.worlds_player = defaultdict(list)  # player -> [(date, win)] at Worlds
        self.elo_hist = defaultdict(list)       # team -> [(date, elo after game)]

    # ---------- setup ----------
    def _league_mean(self, league):
        vals = [self.elo[t] for t, lg in self.league_of.items() if lg == league and t in self.elo]
        return float(np.mean(vals)) if vals else ELO_START

    def ensure(self, team, league, date=None):
        if league not in INTERNATIONAL and league != "unknown":
            self.league_of[team] = league
        if team not in self.elo:
            # New teams start at their league's mean rating.
            self.elo[team] = self._league_mean(league)
        elif date is not None and team in self.last_date and date.year > self.last_date[team].year:
            # New season: partial regression toward the league mean.
            m = self._league_mean(self.league_of.get(team, league))
            self.elo[team] += SEASON_REGRESS * (m - self.elo[team])

    def ensure_players(self, team, lineup):
        for role, p in lineup.items():
            if p not in self.pelo:
                # New players start at their team's current rating.
                self.pelo[p] = self.elo.get(team, ELO_START)

    # ---------- snapshots ----------
    @staticmethod
    def worlds_record(games, date):
        """Recency-weighted Worlds win rate and number of (weighted) games."""
        w_sum = win_sum = 0.0
        for d, win in games:
            w = 0.5 ** (max((date - d).days, 0) / WORLDS_HALF_LIFE)
            w_sum += w
            win_sum += w * win
        return (win_sum + 0.5 * WORLDS_PRIOR) / (w_sum + WORLDS_PRIOR), w_sum

    def _pool(self, p, date):
        dq = self.champs[p]
        while dq and (date - dq[0][0]).days > POOL_DAYS:
            dq.popleft()
        return len({c for _, c in dq})

    def snapshot(self, team, date, lineup=None):
        lineup = lineup if lineup is not None else self.roster.get(team, {})
        s = {"elo": self.elo.get(team, ELO_START), "n_games": self.n[team]}
        for k in FORM_STATS:
            s[f"form_{k}"] = self.form[team].get(k, np.nan)
        dq = self.intl_dates[team]
        s["intl_exp"] = sum(1 for d in dq if (date - d).days <= 365)
        s["worlds_wr"], s["worlds_games"] = self.worlds_record(self.worlds_team[team], date)

        players = [lineup.get(r) for r in ROLES if lineup.get(r)]
        if players:
            s["roster_elo"] = float(np.mean([self.pelo.get(p, s["elo"]) for p in players]))
            for k in PLAYER_FORM:
                vals = [self.pform[p][k] for p in players if k in self.pform[p]]
                s[f"roster_{k}"] = float(np.mean(vals)) if vals else np.nan
            pools = [self._pool(p, date) for p in players]
            s["pool_mean"], s["pool_min"] = float(np.mean(pools)), float(min(pools))
            s["roster_exp"] = float(np.mean([math.log1p(self.pn[p]) for p in players]))
            recs = [self.worlds_record(self.worlds_player[p], date) for p in players]
            s["roster_worlds_wr"] = float(np.mean([r[0] for r in recs]))
            s["roster_worlds_games"] = float(np.mean([r[1] for r in recs]))
        else:
            s["roster_elo"] = s["elo"]
            for k in PLAYER_FORM:
                s[f"roster_{k}"] = np.nan
            s["pool_mean"] = s["pool_min"] = s["roster_exp"] = np.nan
            s["roster_worlds_wr"], s["roster_worlds_games"] = s["worlds_wr"], s["worlds_games"]
        for r in ROLES:
            p = lineup.get(r)
            s[f"role_gd15_{r}"] = self.pform[p].get("gd15", np.nan) if p else np.nan
        return s

    # ---------- updates ----------
    def update(self, row, lineups, player_rows):
        b, r = row["blue"], row["red"]
        k = ELO_K * (INTL_K_MULT if row["league"] in INTERNATIONAL else 1.0) * _mov_mult(row)
        if is_current_worlds(row["league"], row["date"]):
            k *= CURRENT_WORLDS_K_MULT

        # team Elo
        eb, er = self.elo[b], self.elo[r]
        p_blue = 1 / (1 + 10 ** ((er - eb - BLUE_ADV) / 400))
        delta = k * (row["blue_win"] - p_blue)
        self.elo[b], self.elo[r] = eb + delta, er - delta

        # player Elo: expected result from the mean ratings of both lineups
        lb, lr = lineups.get("Blue", {}), lineups.get("Red", {})
        if lb and lr:
            mb = np.mean([self.pelo[p] for p in lb.values()])
            mr = np.mean([self.pelo[p] for p in lr.values()])
            pd_ = k * (row["blue_win"] - 1 / (1 + 10 ** ((mr - mb - BLUE_ADV) / 400)))
            for p in lb.values():
                self.pelo[p] += pd_
            for p in lr.values():
                self.pelo[p] -= pd_

        for team, side in ((b, "blue"), (r, "red")):
            for key, val in _team_stats(row, side).items():
                _ewma(self.form[team], key, val)
            self.n[team] += 1
            self.last_date[team] = row["date"]
            self.elo_hist[team].append((row["date"], self.elo[team]))
            win = row["blue_win"] if side == "blue" else 1 - row["blue_win"]
            if row["league"] in WORLDS:
                self.worlds_team[team].append((row["date"], win))
            if row["league"] in INTERNATIONAL:
                self.intl_dates[team].append(row["date"])
            lu = lineups.get(side.capitalize())
            if lu and len(lu) == 5:
                self.roster[team] = dict(lu)

        for p in player_rows:
            name = p["playername"]
            for key, val in _player_stats(p).items():
                _ewma(self.pform[name], key, val)
            self.pn[name] += 1
            self.player_team[name] = p["teamname"]
            if isinstance(p.get("champion"), str):
                self.champs[name].append((row["date"], p["champion"]))
            if row["league"] in WORLDS and not _nan(p.get("result")):
                self.worlds_player[name].append((row["date"], int(p["result"])))




def _diff_row(sb, sr, game_in_series):
    rec = {}
    for k in sb:
        if k == "n_games":
            continue
        a, b = sb[k], sr[k]
        rec[f"diff_{k}"] = np.nan if _nan(a) or _nan(b) else a - b
    rec["game_in_series"] = game_in_series
    # Under Fearless draft, pool depth should matter more in later games of a series.
    dp = rec.get("diff_pool_mean", np.nan)
    rec["diff_pool_x_game"] = np.nan if _nan(dp) else dp * (game_in_series - 1)
    return rec


def build_features(games, players):
    """Return (features_df, tracker). Features are diffs (blue minus red)
    of snapshots taken *before* each game."""
    tr = Tracker()
    by_game = defaultdict(list)
    if len(players):
        for p in players.to_dict("records"):
            by_game[p["gameid"]].append(p)

    recs = []
    for row in games.to_dict("records"):
        prs = by_game.get(row["gameid"], [])
        lineups = defaultdict(dict)
        for p in prs:
            lineups[p["side"]][p["position"]] = p["playername"]
        tr.ensure(row["blue"], row["league"], row["date"])
        tr.ensure(row["red"], row["league"], row["date"])
        tr.ensure_players(row["blue"], lineups.get("Blue", {}))
        tr.ensure_players(row["red"], lineups.get("Red", {}))

        sb = tr.snapshot(row["blue"], row["date"], lineups.get("Blue"))
        sr = tr.snapshot(row["red"], row["date"], lineups.get("Red"))
        rec = {"gameid": row["gameid"], "date": row["date"], "league": row["league"],
               "blue": row["blue"], "red": row["red"], "blue_win": row["blue_win"],
               "min_games": min(sb["n_games"], sr["n_games"])}
        rec.update(_diff_row(sb, sr, row.get("game_in_series", 1) or 1))
        recs.append(rec)
        tr.update(row, lineups, prs)
    return pd.DataFrame(recs), tr


TEAM_FEATURES = (["diff_elo", "diff_intl_exp", "diff_worlds_wr", "diff_worlds_games"] + [f"diff_form_{k}" for k in FORM_STATS])
PLAYER_FEATURES = (["diff_roster_elo", "diff_roster_exp", "diff_roster_worlds_wr", "diff_roster_worlds_games", "diff_pool_mean", "diff_pool_min",
                    "game_in_series", "diff_pool_x_game"]
                   + [f"diff_roster_{k}" for k in PLAYER_FORM]
                   + [f"diff_role_gd15_{r}" for r in ROLES])
FEATURE_COLS = TEAM_FEATURES + PLAYER_FEATURES


def matchup_features(tracker, team_a, team_b, date, a_is_blue=True, game_in_series=1):
    """Feature row for a future game, using each team's latest roster."""
    sa, sb = tracker.snapshot(team_a, date), tracker.snapshot(team_b, date)
    if not a_is_blue:
        sa, sb = sb, sa
    return _diff_row(sa, sb, game_in_series)
