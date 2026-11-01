"""Unit checks for the simulator. Run: python tests/test_simulate.py"""
import random
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import Series, draw_quarterfinals, series_prob_by_game, series_win_prob, simulate_swiss  # noqa: E402

assert abs(series_win_prob(0.6, 1) - 0.6) < 1e-12
assert abs(series_win_prob(0.6, 3) - (0.6**2 * (3 - 2 * 0.6))) < 1e-12
assert abs(series_win_prob(0.5, 5) - 0.5) < 1e-12
assert series_win_prob(0.6, 5) > series_win_prob(0.6, 3) > 0.6  # longer series favour the better team

teams = [f"T{i}" for i in range(16)]
P = {(a, b, g): 0.5 for a in teams for b in teams if a != b for g in range(1, 6)}
r1 = [(teams[i], teams[i + 8]) for i in range(8)]
rng = random.Random(0)
series = Series(P, {t: 0.0 for t in teams}, rng)
for bo in (1, 3, 5):  # game-by-game formula matches the simple one when p is constant
    assert abs(series_prob_by_game([0.63] * bo, bo) - series_win_prob(0.63, bo)) < 1e-12
for _ in range(2000):
    q, losses, played = simulate_swiss(teams, r1, series, rng)
    assert len(q) == 8
    assert Counter(losses[t] for t in q) == Counter({0: 2, 1: 3, 2: 3})
    qf = draw_quarterfinals(q, losses, played, rng)
    assert sorted(t for m in qf for t in m) == sorted(q)
    zero_loss = [i for i, m in enumerate(qf) if any(losses[t] == 0 for t in m)]
    assert zero_loss == [0, 2]  # the 3-0 teams are in opposite halves
print("all simulator checks passed")

# Partly-known rounds: one round 1 result fixed, the rest simulated.
known = {1: [("T0", "T8", "T8")]}
for _ in range(500):
    q, losses, _ = simulate_swiss(teams, r1, series, rng, known)
    assert len(q) == 8 and losses["T0"] >= 1
print("known-results checks passed")

# Knockouts: probabilities add up, known results are respected.
from simulate import simulate_knockouts  # noqa: E402
qf = [(teams[0], teams[1]), (teams[2], teams[3]), (teams[4], teams[5]), (teams[6], teams[7])]
ko, likely = simulate_knockouts(qf, P, n_sims=4000)
assert abs(ko["champion"].sum() - 1) < 1e-9 and abs(ko["final"].sum() - 2) < 1e-9 and abs(ko["semifinal"].sum() - 4) < 1e-9
ko, _ = simulate_knockouts(qf, P, n_sims=2000, known={frozenset((teams[0], teams[1])): teams[1]})
assert ko.loc[teams[0], "semifinal"] == 0 and ko.loc[teams[1], "semifinal"] == 1
print("knockout checks passed")
