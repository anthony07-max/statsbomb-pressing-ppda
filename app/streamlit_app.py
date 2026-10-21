"""Streamlit explorer: pressing intensity in the 2015/16 Premier League, La Liga, Serie A and Ligue 1.
Run: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from mplsoccer import VerticalPitch

P = Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="Pressing & PPDA explorer", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(P / "team_clusters.csv"), pd.read_csv(P / "team_match.csv"), pd.read_csv(P / "pressure_grid.csv"),
            json.loads((P / "key_findings.json").read_text()))


ts, tm, grid, F = load()
st.title("Pressing intensity explorer: PPDA, pressures and high turnovers")
st.caption("Data: StatsBomb Open Data (2015/16 seasons, 1,517 matches). Built by Anthony Adusei-Poku (github.com/anthony07-max).")
leagues = st.multiselect("Leagues", sorted(ts.league.unique()), sorted(ts.league.unique()))
d = ts[ts.league.isin(leagues)]

t1, t2, t3, t4 = st.tabs(["League table of pressing", "Team pressure map", "Pressing vs chances conceded", "Pressing styles (k-means)"])
with t1:
    metric = st.selectbox("Metric", ["ppda", "pressures_pm", "att_third_pressure_share", "counterpress_share",
                                     "pressure_regain_rate", "high_turnovers_pm", "def_action_height", "np_xg_conceded_pm"])
    asc = metric in ("ppda", "np_xg_conceded_pm")
    s = d.sort_values(metric, ascending=asc)
    st.plotly_chart(px.bar(s, x=metric, y="team", color="league", orientation="h", height=max(400, 18 * len(s)),
                           category_orders={"team": s.team.tolist()}), width="stretch")
with t2:
    team = st.selectbox("Team", sorted(d.team), index=sorted(d.team).index("Liverpool") if "Liverpool" in set(d.team) else 0)
    row = d[d.team == team].iloc[0]
    c = st.columns(4)
    c[0].metric("PPDA", f"{row.ppda:.2f}")
    c[1].metric("Pressures / match", f"{row.pressures_pm:.0f}")
    c[2].metric("Final-third share", f"{100*row.att_third_pressure_share:.1f}%")
    c[3].metric("Style", row["style"])
    g = grid[grid.team == team].groupby(["gx", "gy"]).pressures.sum().unstack(fill_value=0).reindex(index=range(12), columns=range(8), fill_value=0)
    vp = VerticalPitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#efefef", line_zorder=2)
    fig, ax = vp.draw(figsize=(5, 7))
    fig.set_facecolor("#22312b")
    stat = vp.bin_statistic([0], [0], statistic="count", bins=(12, 8)); stat["statistic"] = g.values.T / row.matches
    vp.heatmap(stat, ax=ax, cmap="hot", edgecolors="#22312b", vmin=0, vmax=8)
    ax.set_title(f"{team}: pressures per match (attacking upwards)", color="white")
    st.pyplot(fig, width="content"); plt.close(fig)
    st.subheader("Match by match")
    st.dataframe(tm[tm.team == team][["match_id", "opponent", "ppda", "pressures", "high_turnovers", "np_xg", "np_xg_conceded"]]
                 .round(2), hide_index=True, width="stretch")
with t3:
    y = st.selectbox("Outcome", ["np_xg_conceded_pm", "np_shots_conceded_pm", "high_turnovers_pm", "high_turnover_shots_pm"])
    st.plotly_chart(px.scatter(d, x="ppda", y=y, color="league", hover_name="team", trendline="ols"), width="stretch")
    st.write(f"Across all 80 team-seasons, PPDA vs non-penalty xG conceded per match: r = {F['ppda_vs_xga']['pearson_r']} "
             f"(within-league r = {F['ppda_vs_xga_within_league_r']}). Correlation, not causation: better teams both press more and concede less.")
with t4:
    st.plotly_chart(px.scatter(d, x="pc1", y="pc2", color="style", hover_name="team", hover_data=["league", "ppda", "pressures_pm"],
                               title="Team pressing styles on the first two principal components"), width="stretch")
    st.dataframe(pd.read_csv(P / "cluster_profiles.csv"), hide_index=True, width="stretch")
