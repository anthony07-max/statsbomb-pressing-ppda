"""Turn raw StatsBomb events into tidy per-team-per-match pressing metrics (data/processed/).

Coordinates: StatsBomb pitch is 120 x 80 and every event is recorded from the acting team's
point of view, attacking towards x = 120.

Metric definitions
* PPDA (passes per defensive action) = opponent open-play passes in the opponent's own 60% of the
  pitch (x <= 72 for the passer) / the pressing team's defensive actions in its attacking 60%
  (x >= 48). Defensive actions = tackles, interceptions, fouls committed and challenges
  (StatsBomb "Dribbled Past"). Lower PPDA = more intense pressing.
* High turnover = a possession the team starts in open play (Regular Play / From Counter)
  whose first event is within 40 yards of the opponent's goal (x >= 80).
* Pressure regain = a pressure followed by the same team winning possession within 5 seconds.
* Pitch thirds for pressures: defensive x < 40, middle 40-80, attacking x >= 80.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUES = {"premier_league": "Premier League", "la_liga": "La Liga", "serie_a": "Serie A", "ligue_1": "Ligue 1"}
SET_PIECE_PASS = {"Free Kick", "Corner", "Throw-in", "Goal Kick", "Kick Off"}
OPEN_PLAY = {"Regular Play", "From Counter"}
ZONE_BINS = (12, 8)
# The Ligue 1 file uses two names for the same club in some matches
ALIASES = {"Marseille": "Olympique de Marseille", "Caen": "Stade Malherbe Caen"}  # 10 x 10 yard cells for the pressure grid


def defensive_action(ev):
    return (((ev.type == "Duel") & (ev.subtype == "Tackle")) | ev.type.isin(["Interception", "Foul Committed", "Dribbled Past"]))


def match_metrics(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match. `ev` holds all events of one or more matches."""
    ev = ev.sort_values(["match_id", "index"])
    teams = ev.groupby("match_id").team.unique()
    opp = {(m, t): [o for o in ts if o != t][0] for m, ts in teams.items() for t in ts}
    ev = ev.assign(opponent=[opp[(m, t)] for m, t in zip(ev.match_id, ev.team)])

    passes = ev[(ev.type == "Pass") & ~ev.subtype.isin(SET_PIECE_PASS)]
    opp_pass = passes[passes.x <= 72].groupby(["match_id", "opponent"]).size()  # credited to the pressing team
    da = ev[defensive_action(ev)]
    da_high = da[da.x >= 48].groupby(["match_id", "team"]).size()
    da_height = da.groupby(["match_id", "team"]).x.mean()

    pr = ev[ev.type == "Pressure"]
    third = pd.cut(pr.x, [-1, 40, 80, 121], labels=["def", "mid", "att"])
    pz = pr.groupby(["match_id", "team", third], observed=False).size().unstack(fill_value=0)

    # Possession starts: first event of each possession that belongs to the possession team
    own = ev[ev.team == ev.possession_team]
    starts = own.groupby(["match_id", "possession"]).first().reset_index()
    ht = starts[starts.play_pattern.isin(OPEN_PLAY) & (starts.x >= 80)]
    shots = ev[(ev.type == "Shot")]
    shot_poss = set(zip(shots.match_id, shots.possession))
    ht = ht.assign(shot=[(m, p) in shot_poss for m, p in zip(ht.match_id, ht.possession)])
    ht_n = ht.groupby(["match_id", "team"]).size()
    ht_shot = ht.groupby(["match_id", "team"]).shot.sum()

    # Pressure regains: next possession start by the same team within 5 s, same period
    s = starts[["match_id", "team", "index", "period", "t"]].rename(columns={"index": "s_index", "period": "s_period", "t": "s_t"})
    p = pr[["match_id", "team", "index", "period", "t"]].sort_values("index")
    s = s.sort_values("s_index")
    m = pd.merge_asof(p, s, left_on="index", right_on="s_index", by=["match_id", "team"], direction="forward")
    m["regain"] = (m.s_period == m.period) & ((m.s_t - m.t) <= 5)
    regains = m.groupby(["match_id", "team"]).regain.sum()

    npshot = shots[shots.subtype != "Penalty"]
    xg_for = npshot.groupby(["match_id", "team"]).xg.sum()
    shots_for = npshot.groupby(["match_id", "team"]).size()

    out = pd.DataFrame({"opp_passes": opp_pass, "def_actions_high": da_high, "def_action_height": da_height,
                        "pressures": pr.groupby(["match_id", "team"]).size(),
                        "counterpressures": pr[pr.counterpress].groupby(["match_id", "team"]).size(),
                        "pressure_regains": regains, "high_turnovers": ht_n, "high_turnover_shots": ht_shot,
                        "np_shots": shots_for, "np_xg": xg_for})
    out.index.names = ["match_id", "team"]
    out = out.join(pz.rename(columns=lambda c: f"pressures_{c}")).fillna(0).reset_index()
    out["opponent"] = [opp[(m, t)] for m, t in zip(out.match_id, out.team)]
    conc = out[["match_id", "team", "np_shots", "np_xg"]].rename(columns={"team": "opponent", "np_shots": "np_shots_conceded", "np_xg": "np_xg_conceded"})
    out = out.merge(conc, on=["match_id", "opponent"])
    out["ppda"] = out.opp_passes / out.def_actions_high.replace(0, np.nan)
    return out


def pressure_grid(ev: pd.DataFrame) -> pd.DataFrame:
    """Pressure counts per team on a 12 x 8 grid (10-yard cells)."""
    pr = ev[(ev.type == "Pressure")].copy()
    pr["gx"] = np.clip((pr.x // 10).astype(int), 0, ZONE_BINS[0] - 1)
    pr["gy"] = np.clip((pr.y // 10).astype(int), 0, ZONE_BINS[1] - 1)
    return pr.groupby(["league", "team", "gx", "gy"]).size().rename("pressures").reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tm, grids = [], []
    for key, name in LEAGUES.items():
        ev = pd.read_parquet(RAW / f"{key}.parquet")
        ev["league"] = name
        ev["team"] = ev.team.replace(ALIASES)
        ev["possession_team"] = ev.possession_team.replace(ALIASES)
        mm = match_metrics(ev)
        mm["league"] = name
        tm.append(mm)
        grids.append(pressure_grid(ev))
        print(f"{name}: {ev.match_id.nunique()} matches, {len(mm)} team-matches")
    tm = pd.concat(tm, ignore_index=True)
    tm.to_csv(OUT / "team_match.csv", index=False)
    pd.concat(grids).to_csv(OUT / "pressure_grid.csv", index=False)

    sums = ["opp_passes", "def_actions_high", "pressures", "counterpressures", "pressure_regains", "high_turnovers",
            "high_turnover_shots", "np_shots", "np_xg", "np_shots_conceded", "np_xg_conceded",
            "pressures_def", "pressures_mid", "pressures_att"]
    g = tm.groupby(["league", "team"])
    ts = g[sums].sum()
    ts["matches"] = g.size()
    ts["def_action_height"] = g.def_action_height.mean()
    ts = ts.reset_index()
    ts["ppda"] = ts.opp_passes / ts.def_actions_high
    for c in ["pressures", "counterpressures", "high_turnovers", "high_turnover_shots", "np_shots", "np_xg",
              "np_shots_conceded", "np_xg_conceded", "pressures_att"]:
        ts[f"{c}_pm"] = ts[c] / ts.matches
    ts["att_third_pressure_share"] = ts.pressures_att / ts.pressures
    ts["counterpress_share"] = ts.counterpressures / ts.pressures
    ts["pressure_regain_rate"] = ts.pressure_regains / ts.pressures
    ts["high_turnover_shot_rate"] = ts.high_turnover_shots / ts.high_turnovers
    ts.round(4).to_csv(OUT / "team_season.csv", index=False)
    print(f"{len(ts)} team-seasons written")


if __name__ == "__main__":
    main()
