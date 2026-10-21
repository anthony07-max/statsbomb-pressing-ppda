"""Pressing analysis: league comparison, pressure maps, pressing vs chances conceded, k-means styles.
Writes charts/*.png, data/processed/team_clusters.csv and data/processed/key_findings.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from mplsoccer import Pitch, VerticalPitch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
P, C = ROOT / "data" / "processed", ROOT / "charts"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
LCOL = {"Premier League": "#3d195b", "La Liga": "#ee8707", "Serie A": "#008fd7", "Ligue 1": "#91bf2b"}
CLUSTER_FEATURES = ["ppda", "pressures_pm", "att_third_pressure_share", "counterpress_share",
                    "pressure_regain_rate", "high_turnovers_pm", "def_action_height"]
K = 4
CREDIT = "Data: StatsBomb Open Data, 2015/16"


def save(fig, name):
    fig.savefig(C / name, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)


def name_clusters(cent: pd.DataFrame) -> dict:
    """Label clusters from their standardised centroids (z-scores vs league peers)."""
    rest = list(cent.index)
    dom = cent.loc[rest, "def_action_height"].idxmax(); rest.remove(dom)
    hi = cent.loc[rest, "ppda"].idxmin(); rest.remove(hi)
    deep = cent.loc[rest, "ppda"].idxmax(); rest.remove(deep)
    return {dom: "Dominant high line", hi: "High-intensity press", deep: "Deep block", rest[0]: "Mid-block"}


def main():
    C.mkdir(exist_ok=True)
    ts = pd.read_csv(P / "team_season.csv")
    tm = pd.read_csv(P / "team_match.csv")
    grid = pd.read_csv(P / "pressure_grid.csv")
    ts = ts[ts.matches >= 30].reset_index(drop=True)  # all 80 clubs played 37-38 of their matches in the data
    F = {"team_seasons": len(ts), "matches": int(tm.match_id.nunique()), "team_matches": len(tm)}

    # League-level summary
    lg = tm.groupby("league")[["opp_passes", "def_actions_high", "pressures", "pressures_att", "high_turnovers", "np_xg"]].sum()
    lg["ppda"] = lg.opp_passes / lg.def_actions_high
    lg["pressures_pm"] = tm.groupby("league").pressures.mean()
    lg["att_share"] = lg.pressures_att / lg.pressures
    F["league"] = lg[["ppda", "pressures_pm", "att_share"]].round(3).to_dict(orient="index")

    # 1. PPDA ranking
    r = ts.sort_values("ppda")
    fig, ax = plt.subplots(figsize=(9, 15))
    ax.barh(r.team + " (" + r.league.str.replace("Premier League", "PL") + ")", r.ppda, color=r.league.map(LCOL))
    ax.invert_yaxis(); ax.set_xlabel("PPDA (lower = more intense press)"); ax.tick_params(axis="y", labelsize=7)
    for lgname, col in LCOL.items(): ax.barh([], [], color=col, label=lgname)
    ax.legend(loc="lower right"); ax.set_title("Passes allowed per defensive action, 2015/16\n" + CREDIT, loc="left")
    save(fig, "01_ppda_ranking.png")
    F["lowest_ppda"] = r.head(5)[["team", "league", "ppda"]].round(2).values.tolist()
    F["highest_ppda"] = r.tail(3)[["team", "league", "ppda"]].round(2).values.tolist()

    # 2. League pressure heatmaps (pressures per team-match per 10x10 cell)
    pitch = Pitch(pitch_type="statsbomb", line_zorder=2, pitch_color="#22312b", line_color="#efefef")
    fig, axs = pitch.grid(nrows=2, ncols=2, figheight=9, title_height=.06, endnote_height=.03, axis=False, grid_height=.83)
    fig.set_facecolor("#22312b")
    games = tm.groupby("league").size()
    vmax = 0; mats = {}
    for lgname in LCOL:
        g = grid[grid.league == lgname].groupby(["gx", "gy"]).pressures.sum().unstack(fill_value=0).reindex(index=range(12), columns=range(8), fill_value=0)
        mats[lgname] = g.values.T / games[lgname]; vmax = max(vmax, mats[lgname].max())
    for ax, (lgname, m) in zip(axs["pitch"].flat, mats.items()):
        stat = pitch.bin_statistic([0], [0], statistic="count", bins=(12, 8))
        stat["statistic"] = m
        pitch.heatmap(stat, ax=ax, cmap="hot", vmin=0, vmax=vmax, edgecolors="#22312b")
        pitch.label_heatmap(stat, ax=ax, color="white", fontsize=7, str_format="{:.1f}", ha="center", va="center",
                            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        ax.set_title(f"{lgname}: {lg.loc[lgname, 'pressures_pm']:.0f} pressures/match, {100*lg.loc[lgname, 'att_share']:.1f}% in final third",
                     color="white", fontsize=10)
    axs["title"].text(.5, .5, "Where teams press: pressures per team per match, attacking left to right", color="white", ha="center", va="center", fontsize=14)
    axs["endnote"].text(1, .5, CREDIT, color="#cccccc", ha="right", va="center", fontsize=9)
    save(fig, "02_league_pressure_maps.png")

    # 3. Pressing intensity vs chances conceded (team-season)
    sl = stats.linregress(ts.ppda, ts.np_xg_conceded_pm)
    F["ppda_vs_xga"] = {"pearson_r": round(sl.rvalue, 3), "p_value": float(f"{sl.pvalue:.2g}"), "slope_xga_per_ppda": round(sl.slope, 3)}
    q = ts.ppda.quantile([.25, .75])
    F["xga_pm_top_quartile_press"] = round(ts[ts.ppda <= q[.25]].np_xg_conceded_pm.mean(), 3)
    F["xga_pm_bottom_quartile_press"] = round(ts[ts.ppda >= q[.75]].np_xg_conceded_pm.mean(), 3)
    # Within-league correlation (removes league-wide differences)
    ts["ppda_z"] = ts.groupby("league").ppda.transform(lambda s: (s - s.mean()) / s.std())
    ts["xga_z"] = ts.groupby("league").np_xg_conceded_pm.transform(lambda s: (s - s.mean()) / s.std())
    F["ppda_vs_xga_within_league_r"] = round(ts[["ppda_z", "xga_z"]].corr().iloc[0, 1], 3)
    # Match level: does a team concede less in the matches it presses harder (relative to its own average)?
    mm = tm.dropna(subset=["ppda"]).copy()
    mm = mm[mm.def_actions_high >= 5]
    mm["ppda_dev"] = mm.ppda - mm.groupby("team").ppda.transform("mean")
    mm["xga_dev"] = mm.np_xg_conceded - mm.groupby("team").np_xg_conceded.transform("mean")
    F["match_level_within_team_r"] = round(mm[["ppda_dev", "xga_dev"]].corr().iloc[0, 1], 3)
    fig, ax = plt.subplots(figsize=(9, 6))
    for lgname, g in ts.groupby("league"):
        ax.scatter(g.ppda, g.np_xg_conceded_pm, color=LCOL[lgname], label=lgname, s=40, alpha=.85)
    xs = np.linspace(ts.ppda.min(), ts.ppda.max(), 50); ax.plot(xs, sl.intercept + sl.slope * xs, "k--", lw=1)
    for _, t in ts[(ts.ppda < 8.8) | (ts.ppda > 14.5) | (ts.np_xg_conceded_pm < .7) | (ts.np_xg_conceded_pm > 1.55)].iterrows():
        ax.annotate(t.team, (t.ppda, t.np_xg_conceded_pm), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PPDA (lower = more pressing)"); ax.set_ylabel("Non-penalty xG conceded per match")
    ax.set_title(f"More pressing, fewer chances conceded? r = {sl.rvalue:.2f} across 80 team-seasons\n{CREDIT}", loc="left")
    ax.legend(); save(fig, "03_ppda_vs_xg_conceded.png")

    # 4. High turnovers -> shots
    tot_ht, tot_hts = ts.high_turnovers.sum(), ts.high_turnover_shots.sum()
    F["high_turnovers_total"], F["high_turnover_shot_rate"] = int(tot_ht), round(tot_hts / tot_ht, 3)
    F["most_high_turnovers_pm"] = ts.nlargest(3, "high_turnovers_pm")[["team", "high_turnovers_pm"]].round(2).values.tolist()
    sl2 = stats.linregress(ts.ppda, ts.high_turnovers_pm)
    F["ppda_vs_high_turnovers_r"] = round(sl2.rvalue, 3)
    fig, ax = plt.subplots(figsize=(9, 6))
    for lgname, g in ts.groupby("league"):
        ax.scatter(g.high_turnovers_pm, g.high_turnover_shots / g.matches, color=LCOL[lgname], label=lgname, s=40)
    for _, t in ts.nlargest(6, "high_turnovers_pm").iterrows():
        ax.annotate(t.team, (t.high_turnovers_pm, t.high_turnover_shots / t.matches), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("High turnovers per match (open-play possessions won within 40 yds of goal)")
    ax.set_ylabel("High turnovers ending in a shot, per match"); ax.legend()
    ax.set_title(f"Winning the ball high: {100*F['high_turnover_shot_rate']:.1f}% of high turnovers end in a shot\n{CREDIT}", loc="left")
    save(fig, "04_high_turnovers.png")

    # 5. K-means pressing styles
    # Standardise within each league: raw pressure counts differ a lot between leagues (150 vs 173 per match),
    # so styles are defined relative to league peers rather than simply splitting by league.
    Z = ts.groupby("league")[CLUSTER_FEATURES].transform(lambda s: (s - s.mean()) / s.std())
    X = Z.values
    sil = {k: round(silhouette_score(X, KMeans(k, n_init=50, random_state=0).fit_predict(X)), 3) for k in range(2, 7)}
    km = KMeans(K, n_init=50, random_state=0).fit(X)
    ts["cluster"] = km.labels_
    cent = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    names = name_clusters(cent)
    ts["style"] = ts.cluster.map(names)
    pcs = PCA(2, random_state=0).fit(X); xy = pcs.transform(X)
    ts["pc1"], ts["pc2"] = xy[:, 0], xy[:, 1]
    F["kmeans"] = {"k": K, "silhouette_by_k": sil, "pca_var_explained": [round(v, 3) for v in pcs.explained_variance_ratio_]}
    prof = ts.groupby("style")[CLUSTER_FEATURES + ["np_xg_conceded_pm"]].mean().round(3)
    prof["n_teams"] = ts.groupby("style").size()
    prof["teams"] = ts.sort_values("ppda").groupby("style").team.apply(lambda s: ", ".join(s.head(6)))
    prof.to_csv(P / "cluster_profiles.csv")
    F["clusters"] = prof.reset_index().to_dict(orient="records")
    ts.drop(columns=["ppda_z", "xga_z"]).to_csv(P / "team_clusters.csv", index=False)
    cols = dict(zip(sorted(names.values()), ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    for st_, g in ts.groupby("style"):
        ax.scatter(g.pc1, g.pc2, s=50, color=cols[st_], label=f"{st_} ({len(g)})")
    for _, t in ts.iterrows():
        ax.annotate(t.team, (t.pc1, t.pc2), fontsize=6, xytext=(2, 2), textcoords="offset points", alpha=.8)
    ax.set_xlabel(f"PC1 ({100*pcs.explained_variance_ratio_[0]:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({100*pcs.explained_variance_ratio_[1]:.0f}%)")
    ax.legend(fontsize=8, loc="best"); ax.set_title(f"Pressing styles: k-means (k={K}) on 7 league-standardised pressing metrics, shown on 2 principal components\n{CREDIT}", loc="left")
    save(fig, "05_pressing_style_clusters.png")

    zc = pd.DataFrame(X, columns=CLUSTER_FEATURES).groupby(ts["style"]).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(zc.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(CLUSTER_FEATURES)), ["PPDA", "Pressures\n/match", "Final-third\nshare", "Counterpress\nshare", "Regain\nrate", "High TOs\n/match", "Def. action\nheight"], fontsize=8)
    ax.set_yticks(range(len(zc)), zc.index, fontsize=8)
    for i in range(zc.shape[0]):
        for j in range(zc.shape[1]): ax.text(j, i, f"{zc.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="z-score vs league peers"); ax.set_title("Cluster profiles (standardised means)", loc="left")
    save(fig, "06_cluster_profiles.png")

    # 6. Two contrasting team pressure maps
    hi, lo = r.iloc[0], r.iloc[-1]
    vp = VerticalPitch(pitch_type="statsbomb", line_zorder=2, pitch_color="#22312b", line_color="#efefef")
    fig, axs = vp.grid(nrows=1, ncols=2, figheight=8, title_height=.08, endnote_height=.03, axis=False, grid_height=.82)
    fig.set_facecolor("#22312b")
    for ax, t in zip(axs["pitch"], [hi, lo]):
        g = grid[grid.team == t.team].groupby(["gx", "gy"]).pressures.sum().unstack(fill_value=0).reindex(index=range(12), columns=range(8), fill_value=0)
        stat = vp.bin_statistic([0], [0], statistic="count", bins=(12, 8)); stat["statistic"] = g.values.T / t.matches
        vp.heatmap(stat, ax=ax, cmap="hot", edgecolors="#22312b", vmin=0, vmax=8)
        ax.set_title(f"{t.team}: PPDA {t.ppda:.1f}\n{100*t.att_third_pressure_share:.0f}% of pressures in final third", color="white", fontsize=11)
    axs["title"].text(.5, .5, "Most vs least intense pressing team (pressures per match, attacking upwards)", color="white", ha="center", va="center", fontsize=13)
    axs["endnote"].text(1, .5, CREDIT, color="#cccccc", ha="right", va="center", fontsize=9)
    save(fig, "07_team_pressure_maps.png")

    (P / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
    print(json.dumps(F, indent=2, default=str))


if __name__ == "__main__":
    main()
