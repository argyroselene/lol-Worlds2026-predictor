"""Command-line entry point for the Worlds 2026 knockout predictor.

    python src/run.py --data data --out outputs [--sims 20000] [--focus T1]

Pipeline: load data -> build features -> evaluate models -> refit the best
model -> compute per-team adjustments -> simulate the knockout bracket ->
write CSV/JSON outputs.

Swiss results and the bracket are defined in ``worlds2026.py``. Knockout series
that have already been played can be fixed via ``<data>/knockout_results.csv``
(columns: team_a, team_b, winner).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data import ROLES, load_all
from features import build_features, is_current_worlds
from model import calibration_table, evaluate, feature_importance, fit_final
from simulate import game_prob_matrix, series_prob_by_game, simulate_knockouts
from worlds2026 import QUARTERFINALS, REGION, SWISS_RESULTS, TEAMS, resolve_names, swiss_records


def load_known_ko(path, to_data):
    """Read played knockout series from a CSV with columns team_a, team_b, winner."""
    known = {}
    if Path(path).exists():
        df = pd.read_csv(path, comment="#").fillna("")
        for r in df.itertuples():
            if r.winner.strip():
                known[frozenset((to_data[r.team_a.strip()], to_data[r.team_b.strip()]))] = to_data[r.winner.strip()]
    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--uncertainty", type=float, default=0.25,
                    help="how much a team's true form can differ from its numbers (0 = none)")
    ap.add_argument("--model", default="auto", help="model to use, or 'auto' for the best test score")
    ap.add_argument("--half-life", type=float, default=180,
                    help="days until a game counts half as much in training (0 = all games equal)")
    ap.add_argument("--big-stage", type=float, default=0.5,
                    help="strength of the Worlds track-record boost (0 = off, 1 = strong)")
    ap.add_argument("--this-worlds", type=float, default=1.0,
                    help="strength of the boost from this year's Swiss performance (0 = off, 1 = default, 2 = strong)")
    ap.add_argument("--focus", default="T1", help="team to analyse in detail")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print("1) Loading games...")
    games, players = load_all(args.data)
    print(f"   {len(games):,} games and {players['playername'].nunique() if len(players) else 0:,} "
          f"players, {games.date.min().date()} to {games.date.max().date()}")
    print("   leagues:", ", ".join(games.league.value_counts().index[:30]))

    print("2) Building team and player features...")
    feats, tracker = build_features(games, players)

    print("3) Testing models on the most recent 20% of games...")
    half_life = args.half_life or None
    scores, info, preds, fitted, test = evaluate(feats, half_life=half_life)
    print(f"   train {info['train_games']:,} games, test {info['test_games']:,} games (from {info['test_from']})")
    print(scores.round(4).to_string(index=False))
    best = args.model
    if best == "auto":
        best = scores[scores.model != "always_blue"].sort_values("log_loss").iloc[0]["model"]
    print(f"   using: {best}")
    # Importance is computed on the best full-feature model, regardless of which model is used.
    full = scores[scores.model.isin(["full_logistic", "full_boosting", "ensemble"])]
    imp_model = full.sort_values("log_loss").iloc[0]["model"]
    imp = feature_importance(*fitted[imp_model], test)
    calib = calibration_table(preds["blue_win"], preds[best])
    print(f"   top features ({imp_model}):", ", ".join(imp.feature.head(8)))
    cols, model = fit_final(feats, best, half_life)

    print("4) Simulating the knockout stage...")
    mapping, warnings = resolve_names(set(tracker.elo), tracker.last_date)
    for w in warnings:
        print("   warning:", w)
    to_ours = {v: k for k, v in mapping.items()}
    teams = [mapping[t] for t in TEAMS]
    for t in teams:
        tracker.ensure(t, "unknown")
    as_of = games.date.max()
    cw = games[games.apply(lambda r: is_current_worlds(r["league"], r["date"]), axis=1)]
    print(f"   current-Worlds games in data: {len(cw)}")
    P = game_prob_matrix(teams, tracker, cols, model, as_of)
    records = swiss_records()

    # Worlds track-record offset: scaled log-odds of the roster's weighted Worlds win rate.
    boost_rows, big = [], {}
    for ours, t in mapping.items():
        snap = tracker.snapshot(t, as_of)
        wr = snap["roster_worlds_wr"]
        big[t] = args.big_stage * math.log(wr / (1 - wr))
        boost_rows.append({"team": ours, "roster_worlds_win_rate": wr,
                           "roster_worlds_games_weighted": snap["roster_worlds_games"],
                           "team_worlds_win_rate": snap["worlds_wr"], "boost": big[t]})
    boost_table = pd.DataFrame(boost_rows).set_index("team").sort_values("boost", ascending=False)

    # Current-Worlds offset: sum over Swiss series of (result - model win probability),
    # shrunk by FORM_PRIOR pseudo-series and scaled by --this-worlds.
    form_rows, now = [], {}
    surprise = defaultdict(float)
    n_series = defaultdict(int)
    gw, gl = defaultdict(int), defaultdict(int)
    series_log = []
    for rnd, a, b, winner, bo, score in SWISS_RESULTS:
        pa = series_prob_by_game([P[(mapping[a], mapping[b], g)] for g in range(1, bo + 1)], bo)
        ya = 1.0 if winner == a else 0.0
        surprise[a] += ya - pa
        surprise[b] += (1 - ya) - (1 - pa)
        n_series[a] += 1
        n_series[b] += 1
        if score:
            sa, sb = (int(x) for x in score.split("-"))
            gw[a] += sa; gl[a] += sb; gw[b] += sb; gl[b] += sa
        series_log.append({"round": rnd, "team_a": a, "team_b": b, "best_of": bo, "winner": winner,
                           "score": score or "?", "model_p_a": pa,
                           "upset": (ya - pa) < -0.5 or (ya - pa) > 0.5})
    FORM_PRIOR = 2
    for t in TEAMS:
        adj = surprise[t] / (n_series[t] + FORM_PRIOR)
        now[mapping[t]] = args.this_worlds * 2 * adj
        form_rows.append({"team": t, "swiss_record": records[t], "games_won": gw[t], "games_lost": gl[t],
                          "series_wins": int(records[t].split("-")[0]),
                          "expected_series_wins": int(records[t].split("-")[0]) - surprise[t],
                          "surprise": surprise[t], "this_worlds_boost": now[mapping[t]]})
    form_table = pd.DataFrame(form_rows).set_index("team").sort_values("this_worlds_boost", ascending=False)
    series_log = pd.DataFrame(series_log)
    print("   this-Worlds boost (top 5):", ", ".join(f"{t} {b:+.2f}" for t, b in form_table.this_worlds_boost.head(5).items()))
    print("   big-stage boost (top 5):", ", ".join(f"{t} {b:+.2f}" for t, b in boost_table.boost.head(5).items()))

    qf = [(mapping[a], mapping[b]) for a, b in QUARTERFINALS]
    known = load_known_ko(Path(args.data) / "knockout_results.csv", mapping)
    if known:
        print(f"   knockout results fixed: {len(known)}")
    both = {t: big[t] + now[t] for t in teams}
    runs = {
        "": both,
        "_model_only": None,
        "_no_this_worlds": big,
        "_no_big_stage": now,
    }
    res, likely = None, None
    for suffix, boost in runs.items():
        r, lk = simulate_knockouts(qf, P, n_sims=args.sims, uncertainty=args.uncertainty,
                                   boost=boost, known=known)
        r.index = [to_ours[t] for t in r.index]
        if res is None:
            res, likely = r, lk
        else:
            res[f"champion{suffix}"] = r["champion"]
    likely[["team_1", "team_2"]] = likely[["team_1", "team_2"]].replace(to_ours)

    # Quarterfinal odds: exact model-only probability and simulated probability with offsets.
    qf_rows = []
    for a, b in QUARTERFINALS:
        pa = series_prob_by_game([P[(mapping[a], mapping[b], g)] for g in range(1, 6)], 5)
        qf_rows.append({"team_a": a, "record_a": records[a], "team_b": b, "record_b": records[b],
                        "p_a_wins_model_only": pa, "p_a_wins_with_boosts": res.loc[a, "semifinal"]})
    qf_odds = pd.DataFrame(qf_rows)

    # Team report cards
    cards = []
    for ours, t in mapping.items():
        s = tracker.snapshot(t, as_of)
        roster = tracker.roster.get(t, {})
        cards.append({"team": ours, "region": REGION[ours], "elo": s["elo"], "roster_elo": s["roster_elo"],
                      "win_form": s["form_win"], "gd15_form": s["form_gd15"],
                      "champ_pool": s["pool_mean"], "intl_games_1y": s["intl_exp"],
                      "games_in_data": tracker.n[t], "last_game": str(tracker.last_date.get(t, ""))[:10],
                      **{r: roster.get(r, "") for r in ROLES}})
    cards = pd.DataFrame(cards).set_index("team")

    # Player table for all Worlds rosters
    prows = []
    for ours, t in mapping.items():
        for r, p in tracker.roster.get(t, {}).items():
            f = tracker.pform[p]
            prows.append({"team": ours, "role": r, "player": p, "player_elo": tracker.pelo.get(p),
                          "kda": f.get("kda"), "gd15": f.get("gd15"), "dpm": f.get("dpm"),
                          "vspm": f.get("vspm"), "champ_pool": tracker._pool(p, as_of),
                          "games": tracker.pn[p]})
    player_table = pd.DataFrame(prows)

    table = res.join(cards[["region", "elo", "roster_elo"]])
    table.insert(0, "swiss_record", [records[t] for t in table.index])
    show = (table[["champion", "final", "semifinal", "champion_model_only"]] * 100).round(1)
    print(show.join(table[["swiss_record", "region"]]).to_string())
    print("\nQuarterfinals (best-of-five):")
    print(qf_odds.round(3).to_string(index=False))
    print("\nMost likely finals:")
    print(likely.assign(chance=lambda d: (d.chance * 100).round(1)).to_string(index=False))

    table.to_csv(out / "worlds2026_odds.csv")
    qf_odds.to_csv(out / "quarterfinal_odds.csv", index=False)
    form_table.to_csv(out / "this_worlds_form.csv")
    series_log.to_csv(out / "swiss_series.csv", index=False)
    scores.to_csv(out / "model_scores.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    cards.to_csv(out / "team_report_cards.csv")
    player_table.to_csv(out / "worlds_players.csv", index=False)
    likely.to_csv(out / "likely_finals.csv", index=False)
    names = [t for m in QUARTERFINALS for t in m]
    pm = pd.DataFrame({a: {b: (P[(mapping[a], mapping[b], 1)] if a != b else None) for b in names}
                       for a in names}).T
    pm.to_csv(out / "game_win_matrix.csv")
    bo5 = pd.DataFrame({a: {b: (series_prob_by_game([P[(mapping[a], mapping[b], g)] for g in range(1, 6)], 5)
                                if a != b else None) for b in names} for a in names}).T
    bo5.to_csv(out / "bo5_win_matrix.csv")
    boost_table.to_csv(out / "big_stage_boost.csv")

    # Focus-team outputs
    focus = args.focus
    if focus in mapping:
        ft = mapping[focus]
        hist = []
        for ours, t in mapping.items():
            yrs = defaultdict(lambda: [0, 0])
            for d, w in tracker.worlds_team[t]:
                yrs[d.year][0] += 1
                yrs[d.year][1] += w
            for y, (g, w) in sorted(yrs.items()):
                hist.append({"team": ours, "year": y, "worlds_games": g, "worlds_wins": w,
                             "worlds_win_rate": w / g})
        pd.DataFrame(hist).to_csv(out / "worlds_history.csv", index=False)
        eh = pd.concat([pd.DataFrame(tracker.elo_hist[t], columns=["date", "elo"]).assign(team=ours)
                        for ours, t in mapping.items() if tracker.elo_hist[t]])
        eh.to_csv(out / "elo_history.csv", index=False)
        fp = []
        for r, p in tracker.roster.get(ft, {}).items():
            wr, wg = tracker.worlds_record(tracker.worlds_player[p], as_of)
            games = tracker.worlds_player[p]
            fp.append({"role": r, "player": p, "worlds_games": len(games),
                       "worlds_wins": sum(w for _, w in games), "weighted_worlds_win_rate": wr,
                       "player_elo": tracker.pelo.get(p)})
        pd.DataFrame(fp).to_csv(out / "focus_players_worlds.csv", index=False)
        vs = pd.DataFrame([{"opponent": o,
                            "game_1": P[(ft, mapping[o], 1)],
                            "bo3_no_boost": series_prob_by_game([P[(ft, mapping[o], g)] for g in (1, 2, 3)], 3),
                            "bo5_no_boost": series_prob_by_game([P[(ft, mapping[o], g)] for g in range(1, 6)], 5)}
                           for o in names if o != focus])
        vs.to_csv(out / "focus_vs_teams.csv", index=False)
        if focus in res.index:
            f = res.loc[focus]
            print(f"\n{focus}: win Worlds {f.champion:.1%} (model only {f.champion_model_only:.1%}), "
                  f"reach final {f.final:.1%}, win quarterfinal {f.semifinal:.1%}")

    (out / "run_info.json").write_text(json.dumps({**info, "model": best, "importance_model": imp_model, "focus": args.focus,
                                                   "big_stage": args.big_stage, "this_worlds": args.this_worlds,
                                                   "current_worlds_games_in_csv": int(len(cw)), "as_of": str(as_of.date()),
                                                   "name_warnings": warnings}, indent=2))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
