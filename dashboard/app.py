import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.data_loader import (
    load_race_session,
    get_driver_laps
)

from src.degradation import (
    get_compound_degradation
)

from src.strategy import (
    simulate_strategy_window,
    find_optimal_pit_stop
)

@st.cache_resource
def get_session(race):

    return load_race_session(
        grand_prix=race
    )

st.set_page_config(
    page_title="RaceMind",
    layout="wide"
)

css_path = Path(__file__).resolve().parent.parent / "style.css"

with open(css_path) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.markdown("""
<h1>RaceMind</h1>
<p style="
font-size:24px;
color:#9CA3AF;
margin-top:-20px;">
Formula 1 Strategy Intelligence Platform
</p>
""",
unsafe_allow_html=True)

races = [
    "Abu Dhabi",
    "Bahrain"
]

selected_race = st.selectbox(
    "Select Race",
    races
)

drivers = [
    "VER",
    "NOR",
    "PIA",
    "LEC",
    "RUS"
]

selected_driver = st.selectbox(
    "Select Driver",
    drivers
)

driver_colors = {
    "VER": "#3671C6",
    "NOR": "#FF8000",
    "PIA": "#FF8000",
    "LEC": "#DC0000",
    "RUS": "#00D2BE"
}

accent = driver_colors[selected_driver]

st.markdown(
    f"""
    <style>
    .stSlider {{
        accent-color: {accent};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

pit_lap = st.slider(
    "Pit Stop Lap",
    min_value=18,
    max_value=28,
    value=23
)

st.divider()

session = get_session(
    selected_race
)

st.subheader("Strategy Prediction")

driver_laps = get_driver_laps(
    session,
    selected_driver
)

available_compounds = (
    driver_laps["Compound"]
    .dropna()
    .unique()
)

compound_1 = available_compounds[0]
compound_2 = available_compounds[1]

medium_model = get_compound_degradation(
    driver_laps,
    compound_1
)

hard_model = get_compound_degradation(
    driver_laps,
    compound_2
)

simulation_results = simulate_strategy_window(
    18,
    28,
    medium_model,
    hard_model
)

simulation_df = pd.DataFrame(
    simulation_results
)

best_strategy = find_optimal_pit_stop(
    simulation_df
)

predicted_time = simulation_df[
    simulation_df["PitLap"] == pit_lap
]["PredictedRaceTime"].iloc[0]

left, right = st.columns([2, 1])

with left:

    with st.container(border=True):

        st.markdown(
            f"""
            <h2 style='color:{accent};'>
            {selected_driver}
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
    f"""
    <span style="
    color:#9CA3AF;
    font-size:18px;">
    🏁 {selected_race} Grand Prix
    </span>
    """,
    unsafe_allow_html=True
)

        st.write(
            f"**Tyres:** {' | '.join(map(str, available_compounds))}"
        )

with right:

    st.metric(
        "Race Time",
        f"{predicted_time:.2f}s"
    )

    st.metric(
        "Best Pit Lap",
        str(int(best_strategy["PitLap"]))
    )

st.markdown("### Strategy Simulation")

best_pit_lap = int(
    best_strategy["PitLap"]
)

fig = px.line(
    simulation_df,
    x="PitLap",
    y="PredictedRaceTime",
    markers=True
)

fig.update_traces(
    line_color=accent,
    line_width=5,
    marker_size=10
)

fig.update_layout(
    title="Pit Stop Strategy Simulation",
    paper_bgcolor="#0B1220",
    plot_bgcolor="#0B1220",
    font_color="white",
    title_font_size=24,

    xaxis_title="Pit Stop Lap",
    yaxis_title="Predicted Race Time (s)",

    xaxis=dict(
        showgrid=False
    ),

    yaxis=dict(
        gridcolor="#1F2937"
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

fig.add_vline(
    x=best_pit_lap,
    line_width=3,
    line_dash="dash",
    line_color="#22C55E"
)

fig.add_annotation(
    x=best_pit_lap,
    y=best_strategy["PredictedRaceTime"],
    text=f"BEST LAP {best_pit_lap}",
    showarrow=True,
    arrowhead=2,
    font=dict(
        color="#22C55E",
        size=14
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

time_loss = abs(
    predicted_time -
    best_strategy["PredictedRaceTime"]
)

if pit_lap == int(best_strategy["PitLap"]):

    recommendation = f"""
The current strategy is already optimal.

Pit lap {pit_lap} delivers the lowest predicted race time for {selected_driver}.

No further strategic adjustment is recommended based on tyre degradation.
"""

else:

    recommendation = f"""
The simulation identifies lap {int(best_strategy["PitLap"])} as the optimal pit window.

Keeping the current stop on lap {pit_lap} increases the predicted race time by approximately {time_loss:.2f} seconds.

The earlier tyre switch reduces overall degradation and improves race pace across the second stint.
"""

with st.container(border=True):

    st.subheader("AI Strategy Engineer")

    st.write(recommendation)

st.markdown("### Tyre Degradation Analysis")

st.info(
    "Degradation trends are estimated using historical lap-time regression and may be influenced by fuel load and race conditions."
)

degradation_data = []

for tyre_life in range(1, 31):

    medium_time = (
        medium_model["Intercept"]
        + medium_model["Slope"] * tyre_life
    )

    hard_time = (
        hard_model["Intercept"]
        + hard_model["Slope"] * tyre_life
    )

    degradation_data.append({
        "TyreLife": tyre_life,
        "LapTime": medium_time,
        "Compound": compound_1
    })

    degradation_data.append({
        "TyreLife": tyre_life,
        "LapTime": hard_time,
        "Compound": compound_2
    })

degradation_df = pd.DataFrame(
    degradation_data
)

fig2 = px.line(
    degradation_df,
    x="TyreLife",
    y="LapTime",
    color="Compound",
    markers=True
)

fig2.update_layout(
    paper_bgcolor="#0B1220",
    plot_bgcolor="#0B1220",
    font_color="white",
    title="Tyre Degradation Model",
    xaxis_title="Tyre Life",
    yaxis_title="Predicted Lap Time (s)"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

comparison_data = []

for driver in drivers:

    laps = get_driver_laps(
        session,
        driver
    )

    compounds = (
        laps["Compound"]
        .dropna()
        .unique()
    )

    if len(compounds) < 2:
        continue

    model_1 = get_compound_degradation(
        laps,
        compounds[0]
    )

    model_2 = get_compound_degradation(
        laps,
        compounds[1]
    )

    results = simulate_strategy_window(
        18,
        28,
        model_1,
        model_2
    )

    df = pd.DataFrame(results)

    best = find_optimal_pit_stop(df)

    comparison_data.append({
        "Driver": driver,
        "Best Pit Lap": int(best["PitLap"]),
        "Predicted Time": round(
            best["PredictedRaceTime"],
            2
        )
    })

comparison_df = pd.DataFrame(
    comparison_data
)

comparison_df = comparison_df.sort_values(
    "Predicted Time"
)

leader_time = comparison_df.iloc[0]["Predicted Time"]

comparison_df["Delta"] = (
    comparison_df["Predicted Time"]
    - leader_time
).round(2)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fastest Driver",
        comparison_df.iloc[0]["Driver"]
    )

with col2:
    st.metric(
        "Best Race Time",
        f"{comparison_df.iloc[0]['Predicted Time']:.2f}s"
    )

with col3:
    st.metric(
        "Best Pit Lap",
        int(comparison_df.iloc[0]["Best Pit Lap"])
    )

fastest_driver = comparison_df.iloc[0]

st.success(
    f"Fastest predicted strategy: "
    f"{fastest_driver['Driver']} "
    f"(Pit Lap {fastest_driver['Best Pit Lap']})"
)

with st.container(border=True):

    st.subheader("Winner Prediction")

    winner = comparison_df.iloc[0]
    p2 = comparison_df.iloc[1]

    gap = (
        p2["Predicted Time"]
        - winner["Predicted Time"]
    )

    st.info(
    f"""
Winner Prediction: {winner["Driver"]}

Optimal Pit Window: Lap {winner["Best Pit Lap"]}

Advantage over P2: {gap:.2f} seconds

The simulation predicts {winner["Driver"]} as the strongest strategic contender based on tyre degradation trends and pit-stop optimization.
"""
)

st.markdown("### Driver Comparison")

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

team_colors = {
    "VER": "#3671C6",
    "NOR": "#FF8000",
    "PIA": "#FFB347",
    "LEC": "#DC0000",
    "RUS": "#00D2BE"
}

fig3 = px.scatter(
    comparison_df,
    x="Best Pit Lap",
    y="Predicted Time",
    color="Driver",
    color_discrete_map=team_colors,
    title=f"{selected_race} Strategy Comparison"
)

fig3.update_traces(
    marker=dict(
        size=18,
        line=dict(
            width=2,
            color="white"
        )
    )
)

fig3.update_layout(
    paper_bgcolor="#0B1220",
    plot_bgcolor="#0B1220",
    font_color="white"
)

top_drivers = comparison_df.head(4)

fig3.update_yaxes(
    range=[
        top_drivers["Predicted Time"].min() - 1,
        top_drivers["Predicted Time"].max() + 1
    ]
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.markdown("### Model Insights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Laps Analysed", len(driver_laps))

with c2:
    st.metric("Compounds", len(available_compounds))

with c3:
    st.metric("Window", "18-28")

with c4:
    st.metric("Model", "Linear")