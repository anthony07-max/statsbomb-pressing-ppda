"""Download StatsBomb Open Data events for the 2015/16 Premier League, La Liga, Serie A and Ligue 1 seasons.
(The 2015/16 Bundesliga is only partly released, so it is left out.)

Source: StatsBomb Open Data, https://github.com/statsbomb/open-data (free for non-commercial use,
StatsBomb must be credited). A full event JSON is 3-4 MB per match, so each file is reduced on the
fly to the columns this project needs and saved as one parquet file per league in data/raw/.
Re-running skips leagues that are already downloaded.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
# (competition_id, season_id, label) - every match of each league season is in the open data
COMPETITIONS = [
    (2, 27, "premier_league"),
    (11, 27, "la_liga"),
    (12, 27, "serie_a"),
    (7, 27, "ligue_1"),
]
KEEP = {"Pass", "Shot", "Pressure", "Duel", "Interception", "Foul Committed", "Dribbled Past",
        "Ball Recovery", "Block", "Clearance", "Carry", "Dispossessed", "Miscontrol"}
SESSION = requests.Session()


def get(url):
    err = None
    for _ in range(5):
        try:
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccup: retry
            err = e
    raise err


def reduce_match(match):
    mid = match["match_id"]
    rows = []
    for e in get(f"{BASE}/events/{mid}.json"):
        t = e["type"]["name"]
        if t not in KEEP:
            continue
        sub = e.get(t.lower().replace(" ", "_"), {}) or {}
        loc = e.get("location") or [None, None]
        rows.append({
            "match_id": mid, "index": e["index"], "period": e["period"],
            "t": e["minute"] * 60 + e["second"], "type": t, "team": e["team"]["name"],
            "possession": e["possession"], "possession_team": e["possession_team"]["name"],
            "play_pattern": e["play_pattern"]["name"], "x": loc[0], "y": loc[1],
            "subtype": (sub.get("type") or {}).get("name"), "outcome": (sub.get("outcome") or {}).get("name"),
            "xg": sub.get("statsbomb_xg"), "counterpress": bool(e.get("counterpress", False)),
        })
    return rows


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, sid, label in COMPETITIONS:
        out = RAW / f"{label}.parquet"
        if out.exists():
            print("exists:", out.name)
            continue
        matches = get(f"{BASE}/matches/{cid}/{sid}.json")
        pd.json_normalize(matches).to_csv(RAW / f"{label}_matches.csv", index=False)
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(reduce_match, matches))
        df = pd.DataFrame([r for m in res for r in m])
        df["league"] = label
        df.to_parquet(out, index=False)
        print(f"{label}: {len(matches)} matches, {len(df):,} events", flush=True)


if __name__ == "__main__":
    sys.exit(main())
