"""
app.py — RaceMind
─────────────────
Formula 1 Strategy Intelligence Platform.

Architecture
────────────
  app.py          ← UI layer (this file)
  src/data_loader ← FastF1 session / lap fetching
  src/degradation ← Tyre degradation modelling
  src/strategy    ← Pit-stop simulation engine
  src/prediction  ← High-level verdicts & rankings
"""

import sys
from pathlib import Path

import fastf1
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_race_session,
    get_driver_laps,
    get_race_laps
)
from src.degradation import (
    get_compound_degradation,
    get_stint_summary
)
from src.strategy import (
    simulate_strategy_window,
    find_optimal_pit_stop,
    compare_actual_vs_optimal,
    PIT_STOP_DELTA,
    compare_1stop_vs_2stop
)
from src.prediction import (
    build_strategy_recommendation,
    assess_undercut
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PIT_WINDOW_MIN = 15
PIT_WINDOW_MAX = 35

COMPOUND_COLORS: dict = {
    "SOFT":         "#E8002D",
    "MEDIUM":       "#FFF200",
    "HARD":         "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET":          "#0067FF"
}

DRIVER_COLORS: dict = {
    "VER": "#3671C6", "TSU": "#3671C6",
    "NOR": "#FF8000", "PIA": "#FF8000",
    "LEC": "#DC0000", "HAM": "#DC0000",
    "RUS": "#00D2BE", "ANT": "#00D2BE",
    "ALO": "#006F62", "STR": "#006F62",
    "GAS": "#FF87BC", "DOO": "#FF87BC",
    "OCO": "#B6BABD", "BEA": "#B6BABD",
    "HAD": "#6692FF", "LAW": "#6692FF",
    "ALB": "#005AFF", "SAI": "#005AFF",
    "HUL": "#52E252", "BOR": "#52E252"
}

PLOT_LAYOUT = dict(
    paper_bgcolor="#0B1220",
    plot_bgcolor="#0B1220",
    font_color="white",
    font_family="Inter, sans-serif",
    title_font_size=20,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1
    ),
    margin=dict(l=20, r=20, t=60, b=20)
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RaceMind",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading session data…")
def _get_session(race: str):
    return load_race_session(grand_prix=race)


@st.cache_data(hash_funcs={fastf1.core.Session: id})
def _get_driver_laps_cached(_session, driver: str):
    return get_driver_laps(_session, driver)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — controls
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="rm-sidebar-title">
      <span class="rm-sidebar-eyebrow">CONFIGURATION</span>
      <span class="rm-sidebar-line"></span>
    </div>
    """, unsafe_allow_html=True)

    schedule = fastf1.get_event_schedule(2025)
    races    = schedule["EventName"].tolist()
    if "Pre-Season Testing" in races:
        races.remove("Pre-Season Testing")

    selected_race = st.selectbox("Grand Prix", races, index=0)

    session   = _get_session(selected_race)
    race_laps = get_race_laps(session)

    drivers = sorted(
        session.laps["Driver"].dropna().unique().tolist()
    )

    selected_driver = st.selectbox("Driver", drivers, index=0)

    st.divider()

    pit_lap = st.slider(
        "Simulated Pit Lap",
        min_value=PIT_WINDOW_MIN,
        max_value=PIT_WINDOW_MAX,
        value=(PIT_WINDOW_MIN + PIT_WINDOW_MAX) // 2,
        help="Drag to explore how different pit laps affect predicted race time."
    )

    st.divider()

    gap_to_ahead = st.number_input(
        "Gap to car ahead (s)",
        min_value=0.0,
        max_value=60.0,
        value=2.0,
        step=0.1,
        help="Used for the undercut viability assessment."
    )

    st.markdown(f"""
    <div class="rm-sidebar-stats">
      <div class="rm-sidebar-stat">
        <span class="rm-stat-label">RACE LAPS</span>
        <span class="rm-stat-value">{race_laps}</span>
      </div>
      <div class="rm-sidebar-stat">
        <span class="rm-stat-label">PIT DELTA</span>
        <span class="rm-stat-value">{PIT_STOP_DELTA:.0f}s</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Driver accent colour
accent = DRIVER_COLORS.get(selected_driver, "#3B82F6")

st.markdown(f"""
<style>
.stSlider {{ accent-color: {accent}; }}
div[data-testid="metric-container"] > div:first-child {{
    color: {accent};
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD + VALIDATE DRIVER DATA
# ═══════════════════════════════════════════════════════════════════════════════

driver_laps = _get_driver_laps_cached(session, selected_driver)

available_compounds = (
    driver_laps["Compound"].dropna().unique().tolist()
)

if len(available_compounds) < 2:
    st.error(
        f"Fewer than two compounds found for **{selected_driver}** "
        f"in the {selected_race} Grand Prix.  "
        f"This may be a sprint weekend or an incomplete data load."
    )
    st.stop()

compound_1 = available_compounds[0]
compound_2 = available_compounds[1]

model_1 = get_compound_degradation(driver_laps, compound_1)
model_2 = get_compound_degradation(driver_laps, compound_2)

if model_1 is None or model_2 is None:
    st.error(
        f"Insufficient lap data to fit degradation models for "
        f"**{selected_driver}**.  At least 3 clean laps per compound "
        f"are required."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

simulation_results = simulate_strategy_window(
    PIT_WINDOW_MIN,
    PIT_WINDOW_MAX,
    model_1,
    model_2,
    race_laps=race_laps
)

simulation_df = pd.DataFrame(simulation_results)
best_strategy = find_optimal_pit_stop(simulation_df)
best_pit_lap  = int(best_strategy["PitLap"])

pit_lap_row    = simulation_df[simulation_df["PitLap"] == pit_lap]
predicted_time = (
    pit_lap_row["PredictedRaceTime"].iloc[0]
    if not pit_lap_row.empty
    else best_strategy["PredictedRaceTime"]
)
best_time  = best_strategy["PredictedRaceTime"]
time_delta = abs(predicted_time - best_time)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STRATEGY OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">STRATEGY OVERVIEW</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

left, mid, right = st.columns([2, 1, 1])

with left:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="rm-driver-meta">
              <span class="rm-driver-badge" style="background:{accent};">{selected_driver}</span>
              <span class="rm-grand-prix-tag">{selected_race.upper()}</span>
            </div>
            <h2 class="rm-driver-title" style="color:{accent};">{selected_driver}</h2>
            <p class="rm-driver-sub">
              &#127937; {selected_race} Grand Prix &middot; {race_laps} laps
            </p>
            """,
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            c1_color = COMPOUND_COLORS.get(compound_1.upper(), "#888")
            st.markdown(
                f"""
                <div class="rm-compound-stat">
                  <span class="rm-compound-label" style="color:{c1_color};">&#9679; {compound_1}</span>
                  <span class="rm-compound-value">{model_1['Slope']:+.3f}s/lap</span>
                  <span class="rm-compound-sub">R&sup2; {model_1['R2']:.3f} &middot; N={model_1['LapCount']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_b:
            c2_color = COMPOUND_COLORS.get(compound_2.upper(), "#888")
            st.markdown(
                f"""
                <div class="rm-compound-stat">
                  <span class="rm-compound-label" style="color:{c2_color};">&#9679; {compound_2}</span>
                  <span class="rm-compound-value">{model_2['Slope']:+.3f}s/lap</span>
                  <span class="rm-compound-sub">R&sup2; {model_2['R2']:.3f} &middot; N={model_2['LapCount']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

with mid:
    st.metric(
        "Predicted Race Time",
        f"{predicted_time:.2f}s",
        delta=f"+{time_delta:.2f}s vs optimal" if time_delta > 0.5 else "Optimal ✓",
        delta_color="inverse"
    )
    st.metric("Selected Pit Lap", pit_lap)

with right:
    st.metric("Model-Optimal Pit", best_pit_lap)
    st.metric("Best Predicted Time", f"{best_time:.2f}s")

# ─── CONFIDENCE BAND (NEW) ────────────────────────────────────────────────
# Calculate how flat the optimum is
simulation_df["DeltaFromBest"] = simulation_df["PredictedRaceTime"] - best_time
within_1s = len(simulation_df[simulation_df["DeltaFromBest"] <= 1.0])
within_3s = len(simulation_df[simulation_df["DeltaFromBest"] <= 3.0])
total_window = len(simulation_df)

# Find all pit laps within 1 second of optimal
optimal_window = simulation_df[simulation_df["DeltaFromBest"] <= 1.0]["PitLap"].tolist()
window_range = f"{min(optimal_window)}–{max(optimal_window)}" if optimal_window else f"{best_pit_lap}"

st.caption(
    f"📊 **Strategy window:** {within_1s}/{total_window} pit laps within 1s of optimal "
    f"(Laps {window_range}) | ±{simulation_df['PredictedRaceTime'].std():.2f}s model uncertainty"
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STRATEGY SIMULATION CHART
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">PIT STOP STRATEGY SIMULATION</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=simulation_df["PitLap"],
    y=simulation_df["PredictedRaceTime"],
    mode="lines+markers",
    name="Predicted Race Time",
    line=dict(color=accent, width=3),
    marker=dict(size=7, color=accent),
    hovertemplate=(
        "<b>Pit lap %{x}</b><br>"
        "Predicted time: %{y:.2f}s<extra></extra>"
    )
))

fig.add_trace(go.Scatter(
    x=[pit_lap],
    y=[predicted_time],
    mode="markers",
    name=f"Selected (Lap {pit_lap})",
    marker=dict(size=14, color=accent, symbol="diamond",
                line=dict(color="white", width=2)),
    hovertemplate=(
        f"<b>Selected: Lap {pit_lap}</b><br>"
        f"Time: {predicted_time:.2f}s<extra></extra>"
    )
))

fig.add_trace(go.Scatter(
    x=[best_pit_lap],
    y=[best_time],
    mode="markers",
    name=f"Optimal (Lap {best_pit_lap})",
    marker=dict(size=14, color="#22C55E", symbol="star",
                line=dict(color="white", width=1.5)),
    hovertemplate=(
        f"<b>Optimal: Lap {best_pit_lap}</b><br>"
        f"Time: {best_time:.2f}s<extra></extra>"
    )
))

fig.add_vline(
    x=best_pit_lap,
    line_width=2,
    line_dash="dash",
    line_color="#22C55E",
    annotation_text=f"OPTIMAL LAP {best_pit_lap}",
    annotation_font_color="#22C55E",
    annotation_font_size=12
)
# Add confidence band (±1 second window)
optimal_min_time = best_time
candidates = simulation_df[simulation_df["PredictedRaceTime"] <= optimal_min_time + 1.0]
if not candidates.empty:
    min_lap = candidates["PitLap"].min()
    max_lap = candidates["PitLap"].max()
    
    fig.add_vrect(
        x0=min_lap, x1=max_lap,
        fillcolor="#22C55E",
        opacity=0.08,
        line_width=0,
        annotation_text="±1s window",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#22C55E"
    )

if pit_lap != best_pit_lap:
    fig.add_vline(
        x=pit_lap,
        line_width=2,
        line_dash="dot",
        line_color=accent,
        annotation_text=f"SELECTED LAP {pit_lap}",
        annotation_font_color=accent,
        annotation_font_size=12,
        annotation_position="top left"
    )

fig.update_layout(
    **PLOT_LAYOUT,
    title="1-Stop Strategy — Predicted Race Time by Pit Lap",
    xaxis_title="Pit Stop Lap",
    yaxis_title="Predicted Race Time (s)",
    xaxis=dict(showgrid=False, dtick=2),
    yaxis=dict(gridcolor="#1F2937"),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# In app.py - Add after the 1-stop strategy chart

# ──────────────────────────────────────────────────────────────────────────────
# 2-STOP STRATEGY COMPARISON (NEW FEATURE)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">2-STOP STRATEGY ANALYSIS</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

# Only show if we have enough race laps for 2 stops
if race_laps >= 50:  # Most F1 races are 55-70 laps
    
    with st.spinner("Simulating 2-stop strategies..."):
        stop_comparison = compare_1stop_vs_2stop(
            model_1, model_2, race_laps,
            pit_window_min=PIT_WINDOW_MIN,
            pit_window_max=PIT_WINDOW_MAX
        )
    
    if "Error" not in stop_comparison:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best 1-Stop",
                f"Lap {stop_comparison['OneStop']['PitLap']}",
                f"{stop_comparison['OneStop']['PredictedTime']:.1f}s"
            )
        
        with col2:
            st.metric(
                "Best 2-Stop",
                f"Lap {stop_comparison['TwoStop']['PitLap1']} → {stop_comparison['TwoStop']['PitLap2']}",
                f"{stop_comparison['TwoStop']['PredictedTime']:.1f}s"
            )
        
        with col3:
            gain = stop_comparison['GainFrom2Stop']
            st.metric(
                "2-Stop Advantage",
                f"{gain:+.2f}s" if gain != 0 else "Neutral",
                delta_color="normal" if gain > 0 else "inverse"
            )
        
        recommendation = stop_comparison['Recommended']
        gain_abs = abs(stop_comparison['GainFrom2Stop'])
        
        if gain_abs > 3:
            st.success(
                f"**{recommendation} strategy recommended** — "
                f"would save **{gain_abs:.1f}s** in race time."
            )
        elif gain_abs > 1:
            st.info(
                f"**{recommendation} has marginal advantage** "
                f"({gain_abs:.1f}s). Track position may decide."
            )
        else:
            st.warning(
                "**Strategies are nearly equal** (<1s difference). "
                "Traffic and tyre management will be decisive."
            )
    else:
        st.info("2-stop analysis requires more race laps. Current race too short.")
else:
    st.caption(f"Race length ({race_laps} laps) too short for meaningful 2-stop analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI STRATEGY ENGINEER
# ═══════════════════════════════════════════════════════════════════════════════

recommendation = build_strategy_recommendation(
    driver=selected_driver,
    pit_lap=pit_lap,
    best_pit_lap=best_pit_lap,
    predicted_time=predicted_time,
    best_time=best_time,
    model_1=model_1,
    model_2=model_2
)

with st.container(border=True):
    st.markdown(
        f"""
        <div class="rm-card-header">
          <span class="rm-card-eyebrow">AI STRATEGY ENGINEER</span>
          <span class="rm-card-tag">{selected_driver}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(recommendation)

    actual_vs_optimal = compare_actual_vs_optimal(
        driver_laps, model_1, model_2,
        race_laps=race_laps,
        pit_window_min=PIT_WINDOW_MIN,
        pit_window_max=PIT_WINDOW_MAX
    )

    st.divider()

    col_av1, col_av2, col_av3 = st.columns(3)
    with col_av1:
        st.metric("Actual Pit Lap", actual_vs_optimal["ActualPitLap"])
    with col_av2:
        st.metric("Model-Optimal Pit", actual_vs_optimal["OptimalPitLap"])
    with col_av3:
        delta_val = actual_vs_optimal["TimeDelta"]
        st.metric(
            "Actual vs Optimal",
            f"{delta_val:+.2f}s" if delta_val is not None else "N/A"
        )
    st.caption(actual_vs_optimal["Verdict"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TYRE DEGRADATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">TYRE DEGRADATION ANALYSIS</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

st.info(
    "Lap times are IQR-filtered before fitting to remove safety-car and VSC laps. "
    "Slopes represent seconds lost per lap of tyre age.",
)

deg_tab1, deg_tab2 = st.tabs(["Degradation Model", "Stint Summary"])

with deg_tab1:

    max_tyre_life = race_laps - PIT_WINDOW_MIN + 5
    degradation_data = []

    for tyre_life in range(1, max_tyre_life + 1):
        for model, compound in [(model_1, compound_1), (model_2, compound_2)]:
            degradation_data.append({
                "TyreLife": tyre_life,
                "LapTime":  model["Intercept"] + model["Slope"] * tyre_life,
                "Compound": compound
            })

    degradation_df = (
        pd.DataFrame(degradation_data)
        .sort_values(["Compound", "TyreLife"])
    )

    color_map = {
        c: COMPOUND_COLORS.get(c.upper(), "#888888")
        for c in [compound_1, compound_2]
    }

    # Real lap scatter overlay
    clean_laps = driver_laps.copy()
    clean_laps["LapTimeSeconds"] = clean_laps["LapTime"].dt.total_seconds()
    clean_laps = clean_laps.dropna(
        subset=["TyreLife", "LapTimeSeconds", "Compound"]
    )
    clean_laps = clean_laps[
        clean_laps["Compound"].isin([compound_1, compound_2])
    ]

    fig2 = go.Figure()

    for compound in [compound_1, compound_2]:
        cdata   = clean_laps[clean_laps["Compound"] == compound]
        c_color = COMPOUND_COLORS.get(compound.upper(), "#888888")

        fig2.add_trace(go.Scatter(
            x=cdata["TyreLife"],
            y=cdata["LapTimeSeconds"],
            mode="markers",
            name=f"{compound} (actual)",
            marker=dict(color=c_color, size=6, opacity=0.4,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                f"<b>{compound}</b><br>"
                "Tyre life: %{x}<br>"
                "Lap time: %{y:.3f}s<extra></extra>"
            )
        ))

    for compound in [compound_1, compound_2]:
        mdata   = degradation_df[degradation_df["Compound"] == compound]
        c_color = COMPOUND_COLORS.get(compound.upper(), "#888888")
        model   = model_1 if compound == compound_1 else model_2

        fig2.add_trace(go.Scatter(
            x=mdata["TyreLife"],
            y=mdata["LapTime"],
            mode="lines",
            name=f"{compound} model ({model['Slope']:+.3f}s/lap)",
            line=dict(color=c_color, width=2.5),
            hovertemplate=(
                f"<b>{compound} model</b><br>"
                "Tyre life: %{x}<br>"
                "Predicted: %{y:.3f}s<extra></extra>"
            )
        ))

    fig2.update_layout(
        **PLOT_LAYOUT,
        title="Tyre Degradation — Actual Laps vs Linear Model",
        xaxis_title="Tyre Life (laps)",
        yaxis_title="Lap Time (s)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#1F2937"),
        hovermode="x unified"
    )

    st.plotly_chart(fig2, use_container_width=True)

    r2_1 = model_1["R2"]
    r2_2 = model_2["R2"]
    quality = (
        "Model fit is strong — degradation is highly linear."
        if min(r2_1, r2_2) > 0.7
        else
        "Model fit is moderate.  Non-linear degradation or limited "
        "data may reduce accuracy."
    )
    st.caption(
        f"R² — {compound_1}: **{r2_1:.3f}** | {compound_2}: **{r2_2:.3f}** "
        f"— {quality}"
    )

with deg_tab2:

    stint_summary = get_stint_summary(driver_laps)

    if not stint_summary.empty:
        st.dataframe(
            stint_summary.style.format({
                "AveragePace":      "{:.3f}s",
                "DegradationSlope": "{:+.4f}s/lap"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No stint data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — UNDERCUT VIABILITY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">UNDERCUT VIABILITY</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

undercut = assess_undercut(
    gap_to_ahead=gap_to_ahead,
    pit_lap=pit_lap,
    model_1=model_1,
    model_2=model_2,
    race_laps=race_laps
)

uc_col1, uc_col2, uc_col3 = st.columns(3)

with uc_col1:
    status = "VIABLE" if undercut["Viable"] else "MARGINAL"
    st.metric("Undercut Status", status)

with uc_col2:
    st.metric(
        "Gain per Lap (fresh vs old)",
        f"{undercut['ProjectedGainPerLap']:+.3f}s"
    )

with uc_col3:
    st.metric(
        "Net Projected Gain (3 laps)",
        f"{undercut['TotalProjectedGain']:+.3f}s"
    )

with st.container(border=True):
    st.caption(undercut["Verdict"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DRIVER COMPARISON & WINNER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">DRIVER STRATEGY COMPARISON</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Simulating all drivers…"):
    # Auto-detect baseline driver (fastest in session)
    try:
        results = session.results
        fastest_driver = results.iloc[0]["Abbreviation"]
    except:
        fastest_driver = "VER"
    
    comparison_df = rank_drivers_fair(
        session,
        drivers,
        get_driver_laps_fn=_get_driver_laps_cached,
        get_compound_degradation_fn=get_compound_degradation,
        race_laps=race_laps,
        pit_window_min=PIT_WINDOW_MIN,
        pit_window_max=PIT_WINDOW_MAX,
        baseline_driver=fastest_driver
    )

if comparison_df.empty:
    st.error(
        "Not enough tyre data to rank drivers for this Grand Prix. "
        "Sprint weekends or incomplete sessions may cause this."
    )
    st.stop()

# Show which baseline was used
st.caption(f" Rankings use **{fastest_driver}** as baseline tyre model (fastest driver in session)")

# ── Winner prediction card ────────────────────────────────────────────────────

winner = comparison_df.iloc[0]
p2     = comparison_df.iloc[1] if len(comparison_df) > 1 else None
gap    = (
    round(p2["PredictedTime"] - winner["PredictedTime"], 3)
    if p2 is not None else None
)

with st.container(border=True):
    st.markdown(
        """
        <div class="rm-card-header">
          <span class="rm-card-eyebrow rm-eyebrow-gold"> WINNER PREDICTION</span>
          <span class="rm-card-tag">MODEL OUTPUT</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    wc1, wc2, wc3, wc4 = st.columns(4)

    with wc1:
        w_color = DRIVER_COLORS.get(winner["Driver"], "#3B82F6")
        st.markdown(
            f"""
            <div class="rm-winner-chip">
              <span class="rm-winner-code" style="color:{w_color};">{winner['Driver']}</span>
              <span class="rm-winner-label">PREDICTED WINNER</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with wc2:
        st.metric("Best Race Time", f"{winner['PredictedTime']:.2f}s")

    with wc3:
        st.metric("Optimal Pit Lap", int(winner["OptimalPitLap"]))

    with wc4:
        if gap is not None:
            st.metric("Gap to P2", f"{gap:.3f}s")

    st.divider()
    st.caption(
        f"**{winner['Driver']}** runs "
        f"{winner['Compound1']} → {winner['Compound2']}, "
        f"pitting on lap {int(winner['OptimalPitLap'])}. "
        f"Degradation: {winner['Compound1']} {winner['Deg1']:+.4f}s/lap, "
        f"{winner['Compound2']} {winner['Deg2']:+.4f}s/lap."
    )

# ── Podium strip ──────────────────────────────────────────────────────────────

podium_cols = st.columns(min(3, len(comparison_df)))
medals = ["P1", "P2", "P3"]

for i, col in enumerate(podium_cols):
    row       = comparison_df.iloc[i]
    drv_color = DRIVER_COLORS.get(row["Driver"], "#888")
    with col:
        with st.container(border=True):
            st.markdown(
                f"{medals[i]} "
                f"<span style='color:{drv_color}; font-weight:700;'>"
                f"{row['Driver']}</span>",
                unsafe_allow_html=True
            )
            st.metric("Predicted Time", f"{row['PredictedTime']:.2f}s")
            st.metric("Pit Lap", int(row["OptimalPitLap"]))
            st.caption(
                f"Δ +{row['DeltaToLeader']:.2f}s"
                if row["DeltaToLeader"] > 0 else "Leader"
            )

# ── Strategy scatter ──────────────────────────────────────────────────────────

fig3 = px.scatter(
    comparison_df,
    x="OptimalPitLap",
    y="PredictedTime",
    color="Driver",
    color_discrete_map=DRIVER_COLORS,
    text="Driver",
    title=f"{selected_race} — Strategy Comparison",
    hover_data={
        "Driver":        True,
        "OptimalPitLap":    True,
        "PredictedTime": ":.2f",
        "DeltaToLeader": ":.2f",
        "Compound1":     True,
        "Compound2":     True
    }
)

fig3.update_traces(
    marker=dict(size=16, line=dict(width=1.5, color="white")),
    textposition="top center",
    textfont=dict(size=11, color="white")
)

top_n = comparison_df.head(5)
fig3.update_yaxes(
    range=[
        top_n["PredictedTime"].min() - 2,
        comparison_df["PredictedTime"].max() + 2
    ]
)

fig3.update_layout(
    **PLOT_LAYOUT,
    xaxis_title="Optimal Pit Lap",
    yaxis_title="Predicted Race Time (s)",
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#1F2937"),
    showlegend=False
)

st.plotly_chart(fig3, use_container_width=True)

# ── Stacked bar — stint breakdown ─────────────────────────────────────────────

# Stacked bar chart removed - Stint1Time/Stint2Time not available in fair ranking model

fig4.add_trace(go.Bar(
    name="Stint 2 + pit delta",
    x=comparison_df["Driver"],
    y=comparison_df["Stint2Time"] + PIT_STOP_DELTA,
    marker_color=[
        DRIVER_COLORS.get(d, "#3B82F6")
        for d in comparison_df["Driver"]
    ],
    opacity=0.45,
    hovertemplate="<b>%{x}</b><br>Stint 2 + pit: %{y:.2f}s<extra></extra>"
))

fig4.update_layout(
    **PLOT_LAYOUT,
    barmode="stack",
    title="Race Time Breakdown — Stint 1 vs Stint 2",
    xaxis_title="Driver",
    yaxis_title="Time (s)",
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#1F2937")
)

st.plotly_chart(fig4, use_container_width=True)

# ── Full table + export ───────────────────────────────────────────────────────

st.markdown("""
<div class="rm-subsection-header">FULL DRIVER TABLE</div>
""", unsafe_allow_html=True)

display_df = comparison_df[[
    "Driver", "OptimalPitLap", "PredictedTime",
    "DeltaToLeader", "Compound1", "Compound2", "Deg1", "Deg2"
]].copy()

display_df.columns = [
    "Driver", "Pit Lap", "Predicted Time (s)",
    "Δ Gap (s)", "Compound 1", "Compound 2", 
    "Deg 1 (s/lap)", "Deg 2 (s/lap)"
]

st.dataframe(
    display_df.style
        .format({
            "Predicted Time (s)": "{:.2f}",
            "Δ Gap (s)":          "{:+.2f}",
            "Deg 1 (s/lap)":      "{:+.4f}",
            "Deg 2 (s/lap)":      "{:+.4f}"
        })
        .background_gradient(subset=["Δ Gap (s)"], cmap="RdYlGn_r"),
    use_container_width=True,
    hide_index=True
)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="EXPORT DATA",
    data=csv,
    file_name=f"racemind_{selected_race.replace(' ', '_')}_strategy.csv",
    mime="text/csv"
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="rm-section-header">
  <span class="rm-section-eyebrow">MODEL INSIGHTS</span>
  <span class="rm-section-line"></span>
</div>
""", unsafe_allow_html=True)

mi1, mi2, mi3, mi4, mi5 = st.columns(5)

with mi1:
    st.metric("Laps Analysed", model_1["LapCount"] + model_2["LapCount"])
with mi2:
    st.metric("Compounds", len(available_compounds))
with mi3:
    st.metric("Pit Window", f"{PIT_WINDOW_MIN}–{PIT_WINDOW_MAX}")
with mi4:
    st.metric("Race Laps", race_laps)
with mi5:
    avg_r2 = round((model_1["R2"] + model_2["R2"]) / 2, 3)
    st.metric("Avg Model R²", avg_r2)

st.caption(
    "Degradation modelled via IQR-filtered linear regression (OLS).  "
    "Race time = Σ(stint 1) + pit delta + Σ(stint 2), "
    "evaluated via arithmetic series closed form.  "
    f"Pit stop delta fixed at {PIT_STOP_DELTA:.0f}s."
)