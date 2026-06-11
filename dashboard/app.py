"""
app.py — RaceMind
Formula 1 Strategy Intelligence Platform
"""

import sys
from pathlib import Path

import fastf1
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_race_session, get_driver_laps, get_race_laps
from src.degradation import get_compound_degradation, get_stint_summary
from src.strategy import (
    simulate_strategy_window, find_optimal_pit_stop, compare_actual_vs_optimal,
    PIT_STOP_DELTA, compare_1stop_vs_2stop
)
from src.prediction import build_strategy_recommendation, assess_undercut
from src.ranking import rank_drivers_fair

# ============================================================================
# CONSTANTS
# ============================================================================

PIT_WINDOW_MIN = 15
PIT_WINDOW_MAX = 35

COMPOUND_COLORS = {
    "SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A", "WET": "#0067FF"
}

DRIVER_COLORS = {
    "VER": "#3671C6", "NOR": "#FF8000", "LEC": "#DC0000", "RUS": "#00D2BE",
    "HAM": "#DC0000", "ALO": "#006F62", "SAI": "#005AFF", "PER": "#3671C6",
    "TSU": "#3671C6", "PIA": "#FF8000", "ANT": "#00D2BE", "STR": "#006F62",
    "GAS": "#FF87BC", "OCO": "#B6BABD", "ALB": "#005AFF", "HUL": "#52E252"
}

PLOT_LAYOUT = dict(
    paper_bgcolor="#0B1220", plot_bgcolor="#0B1220", font_color="white",
    font_family="Inter, sans-serif", title_font_size=20,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(l=20, r=20, t=60, b=20)
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="RaceMind", layout="wide", initial_sidebar_state="expanded")

css_path = Path(__file__).resolve().parent.parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="rm-hero">
  <div class="rm-hero-meta">
    <span class="rm-round-badge">STRATEGY ENGINE</span>
    <span class="rm-up-next-badge">&#9654; LIVE MODEL</span>
  </div>
  <h1 class="rm-title">RACE<span class="rm-title-accent">MIND</span></h1>
  <p class="rm-subtitle">Formula 1 Strategy Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CACHED DATA FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner="Loading session data…")
def _get_session(race: str):
    return load_race_session(grand_prix=race)

@st.cache_data(hash_funcs={fastf1.core.Session: id})
def _get_driver_laps_cached(_session, driver: str):
    return get_driver_laps(_session, driver)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class="rm-sidebar-title">
      <span class="rm-sidebar-eyebrow">CONFIGURATION</span>
      <span class="rm-sidebar-line"></span>
    </div>
    """, unsafe_allow_html=True)

    schedule = fastf1.get_event_schedule(2025)
    races = schedule["EventName"].tolist()
    if "Pre-Season Testing" in races:
        races.remove("Pre-Season Testing")

    selected_race = st.selectbox("Grand Prix", races, index=0)
    session = _get_session(selected_race)
    race_laps = get_race_laps(session)
    drivers = sorted(session.laps["Driver"].dropna().unique().tolist())
    selected_driver = st.selectbox("Driver", drivers, index=0)

    st.divider()
    pit_lap = st.slider("Simulated Pit Lap", PIT_WINDOW_MIN, PIT_WINDOW_MAX, 
                        value=(PIT_WINDOW_MIN + PIT_WINDOW_MAX) // 2)
    st.divider()
    gap_to_ahead = st.number_input("Gap to car ahead (s)", 0.0, 60.0, 2.0, 0.1)

    st.markdown(f"""
    <div class="rm-sidebar-stats">
      <div class="rm-sidebar-stat"><span class="rm-stat-label">RACE LAPS</span><span class="rm-stat-value">{race_laps}</span></div>
      <div class="rm-sidebar-stat"><span class="rm-stat-label">PIT DELTA</span><span class="rm-stat-value">{PIT_STOP_DELTA:.0f}s</span></div>
    </div>
    """, unsafe_allow_html=True)

accent = DRIVER_COLORS.get(selected_driver, "#3B82F6")

# ============================================================================
# LOAD DRIVER DATA
# ============================================================================

driver_laps = _get_driver_laps_cached(session, selected_driver)

def get_driver_models(driver_code):
    laps = _get_driver_laps_cached(session, driver_code)
    compounds = laps["Compound"].dropna().unique().tolist()
    if len(compounds) < 2:
        return None, None, None, None
    c1, c2 = compounds[0], compounds[1]
    m1 = get_compound_degradation(laps, c1)
    m2 = get_compound_degradation(laps, c2)
    if m1 is None or m2 is None:
        return None, None, None, None
    return c1, c2, m1, m2

compound_1, compound_2, model_1, model_2 = get_driver_models(selected_driver)

if model_1 is None or model_2 is None:
    st.warning(f"⚠️ **{selected_driver}** doesn't have enough clean laps.")
    fallback_driver = None
    for d in drivers:
        if d == selected_driver:
            continue
        c1, c2, m1, m2 = get_driver_models(d)
        if m1 and m2:
            fallback_driver = d
            compound_1, compound_2, model_1, model_2 = c1, c2, m1, m2
            break
    if fallback_driver:
        st.info(f"📊 Switching to **{fallback_driver}**")
        selected_driver = fallback_driver
        driver_laps = _get_driver_laps_cached(session, selected_driver)
    else:
        st.error(f"No driver with complete data for {selected_race}")
        st.stop()

# ============================================================================
# SIMULATION
# ============================================================================

simulation_results = simulate_strategy_window(PIT_WINDOW_MIN, PIT_WINDOW_MAX, model_1, model_2, race_laps)
simulation_df = pd.DataFrame(simulation_results)
best_strategy = find_optimal_pit_stop(simulation_df)
best_pit_lap = int(best_strategy["PitLap"])
best_time = best_strategy["PredictedRaceTime"]

pit_lap_row = simulation_df[simulation_df["PitLap"] == pit_lap]
predicted_time = pit_lap_row["PredictedRaceTime"].iloc[0] if not pit_lap_row.empty else best_time
time_delta = abs(predicted_time - best_time)

# ============================================================================
# SECTION 1 - STRATEGY OVERVIEW
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">STRATEGY OVERVIEW</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)

left, mid, right = st.columns([2, 1, 1])

with left:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="rm-driver-meta">
              <span class="rm-driver-badge" style="background:{accent};">{selected_race.upper()}</span>
            </div>
            <h1 class="rm-driver-title" style="color:var(--red); font-size:64px; font-weight:800; margin:10px 0 5px 0;">{selected_driver}</h1>
            <p class="rm-driver-sub" style="margin-bottom:20px;">
              🏁 {race_laps} LAPS
            </p>
            """,
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            c1_color = COMPOUND_COLORS.get(compound_1.upper(), "#888")
            st.markdown(f'<div class="rm-compound-stat"><span class="rm-compound-label" style="color:{c1_color};">● {compound_1}</span><span class="rm-compound-value">{model_1["Slope"]:+.3f}s/lap</span><span class="rm-compound-sub">R² {model_1["R2"]:.3f}</span></div>', unsafe_allow_html=True)
        with col_b:
            c2_color = COMPOUND_COLORS.get(compound_2.upper(), "#888")
            st.markdown(f'<div class="rm-compound-stat"><span class="rm-compound-label" style="color:{c2_color};">● {compound_2}</span><span class="rm-compound-value">{model_2["Slope"]:+.3f}s/lap</span><span class="rm-compound-sub">R² {model_2["R2"]:.3f}</span></div>', unsafe_allow_html=True)

with mid:
    st.metric("Predicted Race Time", f"{predicted_time:.2f}s", delta=f"+{time_delta:.2f}s vs optimal" if time_delta > 0.5 else "Optimal ✓", delta_color="inverse")
    st.metric("Selected Pit Lap", pit_lap)

with right:
    st.metric("Model-Optimal Pit", best_pit_lap)
    st.metric("Best Predicted Time", f"{best_time:.2f}s")

# ============================================================================
# SECTION 2 - STRATEGY CHART
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">PIT STOP STRATEGY SIMULATION</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=simulation_df["PitLap"], y=simulation_df["PredictedRaceTime"], mode="lines+markers", name="Predicted Race Time", line=dict(color=accent, width=3), marker=dict(size=7, color=accent)))
fig.add_trace(go.Scatter(x=[pit_lap], y=[predicted_time], mode="markers", name=f"Selected (Lap {pit_lap})", marker=dict(size=14, color=accent, symbol="diamond", line=dict(color="white", width=2))))
fig.add_trace(go.Scatter(x=[best_pit_lap], y=[best_time], mode="markers", name=f"Optimal (Lap {best_pit_lap})", marker=dict(size=14, color="#22C55E", symbol="star", line=dict(color="white", width=1.5))))
fig.add_vline(x=best_pit_lap, line_width=2, line_dash="dash", line_color="#22C55E", annotation_text=f"OPTIMAL LAP {best_pit_lap}", annotation_font_color="#22C55E", annotation_font_size=12)

candidates = simulation_df[simulation_df["PredictedRaceTime"] <= best_time + 1.0]
if not candidates.empty:
    fig.add_vrect(x0=candidates["PitLap"].min(), x1=candidates["PitLap"].max(), fillcolor="#22C55E", opacity=0.08, line_width=0, annotation_text="±1s window", annotation_position="top left", annotation_font_size=10, annotation_font_color="#22C55E")
if pit_lap != best_pit_lap:
    fig.add_vline(x=pit_lap, line_width=2, line_dash="dot", line_color=accent, annotation_text=f"SELECTED LAP {pit_lap}", annotation_font_color=accent, annotation_font_size=12, annotation_position="top left")

fig.update_layout(**PLOT_LAYOUT, title="1-Stop Strategy — Predicted Race Time by Pit Lap", xaxis_title="Pit Stop Lap", yaxis_title="Predicted Race Time (s)", xaxis=dict(showgrid=False, dtick=2), yaxis=dict(gridcolor="#1F2937"), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 2-STOP COMPARISON
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">2-STOP STRATEGY ANALYSIS</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)

if race_laps >= 50:
    with st.spinner("Simulating 2-stop strategies..."):
        stop_comparison = compare_1stop_vs_2stop(model_1, model_2, race_laps, PIT_WINDOW_MIN, PIT_WINDOW_MAX)
    if "Error" not in stop_comparison:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best 1-Stop", f"Lap {stop_comparison['OneStop']['PitLap']}", f"{stop_comparison['OneStop']['PredictedTime']:.1f}s")
        col2.metric("Best 2-Stop", f"Lap {stop_comparison['TwoStop']['PitLap1']} → {stop_comparison['TwoStop']['PitLap2']}", f"{stop_comparison['TwoStop']['PredictedTime']:.1f}s")
        gain = stop_comparison['GainFrom2Stop']
        col3.metric("2-Stop Advantage", f"{gain:+.2f}s" if gain != 0 else "Neutral", delta_color="normal" if gain > 0 else "inverse")
        if abs(gain) > 3:
            st.success(f"**{stop_comparison['Recommended']} strategy recommended** — would save **{abs(gain):.1f}s**")
        elif abs(gain) > 1:
            st.info(f"**{stop_comparison['Recommended']} has marginal advantage** ({abs(gain):.1f}s)")
        else:
            st.warning("**Strategies are nearly equal** (<1s difference)")
    else:
        st.info("2-stop analysis requires more race laps")
else:
    st.caption(f"Race length ({race_laps} laps) too short for 2-stop analysis")

# ============================================================================
# SECTION 3 - AI STRATEGY ENGINEER
# ============================================================================

recommendation = build_strategy_recommendation(selected_driver, pit_lap, best_pit_lap, predicted_time, best_time, model_1, model_2)
with st.container(border=True):
    st.markdown(f'<div class="rm-card-header"><span class="rm-card-eyebrow">AI STRATEGY ENGINEER</span><span class="rm-card-tag">{selected_driver}</span></div>', unsafe_allow_html=True)
    st.write(recommendation)
    actual_vs_optimal = compare_actual_vs_optimal(driver_laps, model_1, model_2, race_laps, PIT_WINDOW_MIN, PIT_WINDOW_MAX)
    st.divider()
    col_av1, col_av2, col_av3 = st.columns(3)
    col_av1.metric("Actual Pit Lap", actual_vs_optimal["ActualPitLap"])
    col_av2.metric("Model-Optimal Pit", actual_vs_optimal["OptimalPitLap"])
    delta_val = actual_vs_optimal["TimeDelta"]
    col_av3.metric("Actual vs Optimal", f"{delta_val:+.2f}s" if delta_val else "N/A")
    st.caption(actual_vs_optimal["Verdict"])

# ============================================================================
# SECTION 4 - TYRE DEGRADATION
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">TYRE DEGRADATION ANALYSIS</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)
st.info("Lap times are IQR-filtered to remove safety-car and VSC laps.")

deg_tab1, deg_tab2 = st.tabs(["Degradation Model", "Stint Summary"])

with deg_tab1:
    max_tyre_life = race_laps - PIT_WINDOW_MIN + 5
    degradation_data = []
    for tyre_life in range(1, max_tyre_life + 1):
        for model, compound in [(model_1, compound_1), (model_2, compound_2)]:
            degradation_data.append({"TyreLife": tyre_life, "LapTime": model["Intercept"] + model["Slope"] * tyre_life, "Compound": compound})
    degradation_df = pd.DataFrame(degradation_data).sort_values(["Compound", "TyreLife"])
    
    clean_laps = driver_laps.copy()
    clean_laps["LapTimeSeconds"] = clean_laps["LapTime"].dt.total_seconds()
    clean_laps = clean_laps.dropna(subset=["TyreLife", "LapTimeSeconds", "Compound"])
    clean_laps = clean_laps[clean_laps["Compound"].isin([compound_1, compound_2])]
    
    fig2 = go.Figure()
    for compound in [compound_1, compound_2]:
        cdata = clean_laps[clean_laps["Compound"] == compound]
        c_color = COMPOUND_COLORS.get(compound.upper(), "#888888")
        fig2.add_trace(go.Scatter(x=cdata["TyreLife"], y=cdata["LapTimeSeconds"], mode="markers", name=f"{compound} (actual)", marker=dict(color=c_color, size=6, opacity=0.4)))
    
    for compound in [compound_1, compound_2]:
        mdata = degradation_df[degradation_df["Compound"] == compound]
        c_color = COMPOUND_COLORS.get(compound.upper(), "#888888")
        model = model_1 if compound == compound_1 else model_2
        fig2.add_trace(go.Scatter(x=mdata["TyreLife"], y=mdata["LapTime"], mode="lines", name=f"{compound} model ({model['Slope']:+.3f}s/lap)", line=dict(color=c_color, width=2.5)))
    
    fig2.update_layout(**PLOT_LAYOUT, title="Tyre Degradation — Actual Laps vs Linear Model", xaxis_title="Tyre Life (laps)", yaxis_title="Lap Time (s)", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1F2937"), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""<div class="rm-card-header"><span class="rm-card-eyebrow">MODEL SELECTION</span><span class="rm-card-tag">LINEAR REGRESSION</span></div>""", unsafe_allow_html=True)
    st.caption("""**Why linear?** Pirelli tyre degradation is approximately linear until the performance cliff (last 3-5 laps). Linear regression avoids overfitting and provides interpretable "seconds per lap" rates.\n\n**R² interpretation:** >0.7 → Strong | 0.3-0.7 → Moderate | <0.3 → Noisy data""")
    st.caption(f"R² — {compound_1}: **{model_1['R2']:.3f}** | {compound_2}: **{model_2['R2']:.3f}**")

with deg_tab2:
    stint_summary = get_stint_summary(driver_laps)
    if not stint_summary.empty:
        st.dataframe(stint_summary.style.format({"AveragePace": "{:.3f}s", "DegradationSlope": "{:+.4f}s/lap"}), use_container_width=True, hide_index=True)
    else:
        st.warning("No stint data available")

# ============================================================================
# SECTION 5 - UNDERCUT
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">UNDERCUT VIABILITY</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)

undercut = assess_undercut(gap_to_ahead, pit_lap, model_1, model_2, race_laps)
uc_col1, uc_col2, uc_col3 = st.columns(3)
uc_col1.metric("Undercut Status", "VIABLE" if undercut["Viable"] else "MARGINAL")
uc_col2.metric("Gain per Lap", f"{undercut['ProjectedGainPerLap']:+.3f}s")
uc_col3.metric("Net Gain (3 laps)", f"{undercut['TotalProjectedGain']:+.3f}s")
with st.container(border=True):
    st.caption(undercut["Verdict"])

# ============================================================================
# SECTION 6 - DRIVER COMPARISON
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">DRIVER STRATEGY COMPARISON</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)

with st.spinner("Simulating all drivers…"):
    try:
        fastest_driver = session.results.iloc[0]["Abbreviation"]
    except:
        fastest_driver = "VER"
    comparison_df = rank_drivers_fair(session, drivers, _get_driver_laps_cached, get_compound_degradation, race_laps, PIT_WINDOW_MIN, PIT_WINDOW_MAX, fastest_driver)

if comparison_df.empty:
    st.error("Not enough tyre data to rank drivers")
    st.stop()

st.caption(f"⚡ Rankings use **{fastest_driver}** as baseline tyre model")

winner = comparison_df.iloc[0]
p2 = comparison_df.iloc[1] if len(comparison_df) > 1 else None
gap = round(p2["PredictedTime"] - winner["PredictedTime"], 3) if p2 is not None else None

with st.container(border=True):
    st.markdown("""<div class="rm-card-header"><span class="rm-card-eyebrow rm-eyebrow-gold">WINNER PREDICTION</span><span class="rm-card-tag">MODEL OUTPUT</span></div>""", unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    w_color = DRIVER_COLORS.get(winner["Driver"], "#3B82F6")
    wc1.markdown(f'<div class="rm-winner-chip"><span class="rm-winner-code" style="color:{w_color};">{winner["Driver"]}</span><span class="rm-winner-label">PREDICTED WINNER</span></div>', unsafe_allow_html=True)
    wc2.metric("Best Race Time", f"{winner['PredictedTime']:.2f}s")
    wc3.metric("Optimal Pit Lap", int(winner["OptimalPitLap"]))
    if gap:
        wc4.metric("Gap to P2", f"{gap:.3f}s")
    st.caption(f"**{winner['Driver']}** runs {winner['Compound1']} → {winner['Compound2']}, pitting on lap {int(winner['OptimalPitLap'])}")

# Podium
podium_cols = st.columns(min(3, len(comparison_df)))
for i, col in enumerate(podium_cols):
    row = comparison_df.iloc[i]
    drv_color = DRIVER_COLORS.get(row["Driver"], "#888")
    with col:
        with st.container(border=True):
            st.markdown(f"{['P1','P2','P3'][i]} <span style='color:{drv_color}; font-weight:700;'>{row['Driver']}</span>", unsafe_allow_html=True)
            st.metric("Predicted Time", f"{row['PredictedTime']:.2f}s")
            st.metric("Pit Lap", int(row["OptimalPitLap"]))
            st.caption(f"Δ +{row['DeltaToLeader']:.2f}s" if row["DeltaToLeader"] > 0 else "Leader")

# Strategy scatter
fig3 = px.scatter(comparison_df, x="OptimalPitLap", y="PredictedTime", color="Driver", color_discrete_map=DRIVER_COLORS, text="Driver", title=f"{selected_race} — Strategy Comparison", hover_data={"Driver": True, "OptimalPitLap": True, "PredictedTime": ":.2f", "DeltaToLeader": ":.2f", "Compound1": True, "Compound2": True})
fig3.update_traces(marker=dict(size=16, line=dict(width=1.5, color="white")), textposition="top center", textfont=dict(size=11, color="white"))
fig3.update_layout(**PLOT_LAYOUT, xaxis_title="Optimal Pit Lap", yaxis_title="Predicted Race Time (s)", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1F2937"), showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# Full table
st.markdown("""<div class="rm-subsection-header">FULL DRIVER TABLE</div>""", unsafe_allow_html=True)
display_df = comparison_df[["Driver", "OptimalPitLap", "PredictedTime", "DeltaToLeader", "Compound1", "Compound2", "Deg1", "Deg2"]].copy()
display_df.columns = ["Driver", "Pit Lap", "Predicted Time (s)", "Δ Gap (s)", "Compound 1", "Compound 2", "Deg 1 (s/lap)", "Deg 2 (s/lap)"]
st.dataframe(display_df.style.format({"Predicted Time (s)": "{:.2f}", "Δ Gap (s)": "{:+.2f}", "Deg 1 (s/lap)": "{:+.4f}", "Deg 2 (s/lap)": "{:+.4f}"}).background_gradient(subset=["Δ Gap (s)"], cmap="RdYlGn_r"), use_container_width=True, hide_index=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("EXPORT DATA", csv, f"racemind_{selected_race.replace(' ', '_')}_strategy.csv", "text/csv")

# ============================================================================
# SECTION 7 - MODEL INSIGHTS
# ============================================================================

st.markdown("""<div class="rm-section-header"><span class="rm-section-eyebrow">MODEL INSIGHTS</span><span class="rm-section-line"></span></div>""", unsafe_allow_html=True)
mi1, mi2, mi3, mi4, mi5 = st.columns(5)
mi1.metric("Laps Analysed", model_1["LapCount"] + model_2["LapCount"])
mi2.metric("Compounds", len([compound_1, compound_2]))
mi3.metric("Pit Window", f"{PIT_WINDOW_MIN}–{PIT_WINDOW_MAX}")
mi4.metric("Race Laps", race_laps)
mi5.metric("Avg Model R²", round((model_1["R2"] + model_2["R2"]) / 2, 3))
st.caption(f"Degradation via IQR-filtered linear regression. Pit delta: {PIT_STOP_DELTA:.0f}s")