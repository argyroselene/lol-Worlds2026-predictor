"""Monte Carlo tournament simulation.

The model provides P(a beats b) for each game number of a series (game number
matters under Fearless draft). Series are played game by game.

Each simulated tournament draws a random strength offset per team
(``uncertainty``, in log-odds) to reflect form that the model cannot see.
Optional fixed per-team offsets (``boost``) add the Worlds track-record and
current-Worlds adjustments computed in ``run.py``.

``simulate_knockouts`` covers the single-elimination bracket. The Swiss-stage
simulator (``simulate_tournament``) is kept for earlier stages and tests.
"""
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

import pandas as pd

from features import matchup_features

MAX_GAMES = 5


def series_win_prob(p, best_of):
    """P(win a best-of-N series) when every game is won with probability p."""
    need = best_of // 2 + 1
    return sum(comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def series_prob_by_game(probs, best_of):
    """P(win a best-of-N series) when game g is won with probability probs[g-1]."""
    need = best_of // 2 + 1
    # state: (my wins, their wins) -> probability
    states = {(0, 0): 1.0}
    total = 0.0
    for g in range(best_of):
        nxt = defaultdict(float)
        for (w, l), pr in states.items():
            p = probs[g]
            for (dw, dl, q) in ((1, 0, p), (0, 1, 1 - p)):
                s = (w + dw, l + dl)
                if s[0] == need:
                    total += pr * q
                elif s[1] < need:
                    nxt[s] += pr * q
        states = nxt
    return total


def game_prob_matrix(teams, tracker, cols, model, date):
    """P[(a, b, g)] = P(a beats b in game g of a series), averaged over both sides."""
    rows, keys = [], []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            for g in range(1, MAX_GAMES + 1):
                rows.append(matchup_features(tracker, a, b, date, True, g))
                rows.append(matchup_features(tracker, a, b, date, False, g))
                keys.append((a, b, g))
    X = pd.DataFrame(rows).reindex(columns=cols)
    p_blue = model.predict_proba(X)[:, 1]
    P = {}
    for i, k in enumerate(keys):
        P[k] = 0.5 * (p_blue[2 * i] + 1 - p_blue[2 * i + 1])
    for a, b in combinations(teams, 2):  # make P(a,b) + P(b,a) = 1 exactly
        for g in range(1, MAX_GAMES + 1):
            avg = 0.5 * (P[(a, b, g)] + 1 - P[(b, a, g)])
            P[(a, b, g)], P[(b, a, g)] = avg, 1 - avg
    return P


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class Series:
    """Plays series game by game using per-team log-odds offsets."""

    def __init__(self, P, nudge, rng):
        self.P, self.nudge, self.rng = P, nudge, rng

    def game_p(self, a, b, g):
        z = _logit(self.P[(a, b, g)]) + self.nudge[a] - self.nudge[b]
        return 1 / (1 + math.exp(-z))

    def play(self, a, b, best_of):
        need = best_of // 2 + 1
        wa = wb = 0
        g = 1
        while wa < need and wb < need:
            if self.rng.random() < self.game_p(a, b, g):
                wa += 1
            else:
                wb += 1
            g += 1
        return a if wa == need else b


def _pair_group(group, played, rng, tries=200):
    """Randomly pair teams with the same record, avoiding rematches where possible."""
    for _ in range(tries):
        g = group[:]
        rng.shuffle(g)
        pairs = [(g[i], g[i + 1]) for i in range(0, len(g), 2)]
        if all(frozenset(p) not in played for p in pairs):
            return pairs
    return pairs


def simulate_swiss(teams, round1, series, rng, known=None):
    """Swiss stage: three wins advance, three losses eliminate.

    Matches that can advance or eliminate a team are Bo3, others Bo1.
    ``known`` maps round -> list of (a, b, winner or None) already drawn or played.
    """
    known = known or {}
    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    played = set()
    rnd = 1
    while True:
        active = [t for t in teams if wins[t] < 3 and losses[t] < 3]
        if not active:
            break
        # Matches already drawn/played this round come first; everyone
        # else is paired by the round 1 draw or randomly within their record.
        matches = [(a, b) for a, b, _ in known.get(rnd, [])]
        fixed = {frozenset((a, b)): w for a, b, w in known.get(rnd, [])}
        busy = {t for m in matches for t in m}
        if rnd == 1:
            matches += [m for m in round1 if not busy & set(m)]
        else:
            groups = defaultdict(list)
            for t in active:
                if t not in busy:
                    groups[(wins[t], losses[t])].append(t)
            for g in groups.values():
                matches += _pair_group(g, played, rng)
        for a, b in matches:
            decisive = rnd > 1 and (wins[a] == 2 or losses[a] == 2)
            w = fixed.get(frozenset((a, b))) or series.play(a, b, 3 if decisive else 1)
            l = b if w == a else a
            wins[w] += 1
            losses[l] += 1
            played.add(frozenset((a, b)))
        rnd += 1
    qualified = [t for t in teams if wins[t] == 3]
    return qualified, losses, played


def draw_quarterfinals(qualified, losses, played, rng):
    """Quarterfinal draw: 3-0 teams in opposite halves facing 3-2 teams; the rest
    drawn randomly, avoiding Swiss rematches where possible."""
    r30 = [t for t in qualified if losses[t] == 0]
    rest = [t for t in qualified if losses[t] > 0]
    r32 = [t for t in rest if losses[t] == 2]
    for _ in range(200):
        rng.shuffle(r32)
        opp30 = r32[:len(r30)]
        others = [t for t in rest if t not in opp30]
        rng.shuffle(others)
        qf = list(zip(r30, opp30)) + [(others[i], others[i + 1]) for i in range(0, len(others), 2)]
        if all(frozenset(m) not in played for m in qf):
            break
    # Bracket order: QF0 v QF1 in one semi, QF2 v QF3 in the other,
    # so the two 3-0 teams (placed at 0 and 2) can only meet in the final.
    return [qf[0], qf[2], qf[1], qf[3]]


def simulate_tournament(teams, round1, P, n_sims=20000, seed=0, known=None,
                        uncertainty=0.25, region=None, boost=None):
    rng = random.Random(seed)
    stages = ("quarterfinal", "semifinal", "final", "champion")
    counts = {k: Counter() for k in stages}
    records = Counter()          # (team, swiss record) e.g. ("T1", "3-1")
    finals = Counter()           # most likely final matchups
    region_wins = Counter()
    for _ in range(n_sims):
        boost = boost or {}
        nudge = {t: boost.get(t, 0.0) + rng.gauss(0, uncertainty) for t in teams}
        series = Series(P, nudge, rng)
        qualified, losses, played = simulate_swiss(teams, round1, series, rng, known)
        for t in qualified:
            records[(t, f"3-{losses[t]}")] += 1
        qf = draw_quarterfinals(qualified, losses, played, rng)
        counts["quarterfinal"].update(qualified)
        semis = [series.play(a, b, 5) for a, b in qf]
        counts["semifinal"].update(semis)
        f = [series.play(semis[0], semis[1], 5), series.play(semis[2], semis[3], 5)]
        counts["final"].update(f)
        finals[tuple(sorted(f))] += 1
        champ = series.play(f[0], f[1], 5)
        counts["champion"][champ] += 1
        if region:
            region_wins[region.get(champ, "?")] += 1
    out = pd.DataFrame({k: {t: counts[k][t] / n_sims for t in teams} for k in stages})
    for rec in ("3-0", "3-1", "3-2"):
        out[f"swiss_{rec}"] = [records[(t, rec)] / n_sims for t in out.index]
    out = out.sort_values(["champion", "final", "semifinal"], ascending=False)
    extra = {
        "finals": pd.DataFrame([{"team_1": a, "team_2": b, "chance": c / n_sims}
                                for (a, b), c in finals.most_common(10)]),
        "regions": pd.Series({k: v / n_sims for k, v in region_wins.most_common()}, name="champion"),
    }
    return out, extra


def simulate_knockouts(quarterfinals, P, n_sims=20000, seed=0, uncertainty=0.25,
                       boost=None, known=None):
    """Single-elimination bracket of best-of-five series.

    ``quarterfinals`` is in bracket order: winners of QF1/QF2 and QF3/QF4 meet
    in the semifinals. ``known`` maps frozenset({a, b}) -> winner for series
    already played.
    """
    rng = random.Random(seed)
    boost = boost or {}
    known = known or {}
    teams = [t for m in quarterfinals for t in m]
    stages = ("semifinal", "final", "champion")
    counts = {k: Counter() for k in stages}
    finals = Counter()

    for _ in range(n_sims):
        nudge = {t: boost.get(t, 0.0) + rng.gauss(0, uncertainty) for t in teams}
        series = Series(P, nudge, rng)

        def play(a, b):
            return known.get(frozenset((a, b))) or series.play(a, b, 5)

        semis = [play(a, b) for a, b in quarterfinals]
        counts["semifinal"].update(semis)
        f = [play(semis[0], semis[1]), play(semis[2], semis[3])]
        counts["final"].update(f)
        finals[tuple(sorted(f))] += 1
        counts["champion"][play(f[0], f[1])] += 1

    out = pd.DataFrame({k: {t: counts[k][t] / n_sims for t in teams} for k in stages})
    out = out.sort_values(["champion", "final", "semifinal"], ascending=False)
    likely = pd.DataFrame([{"team_1": a, "team_2": b, "chance": c / n_sims}
                           for (a, b), c in finals.most_common(10)])
    return out, likely
