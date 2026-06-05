import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.strategy import (
    simulate_strategy_window,
    find_optimal_pit_stop
)

import pandas as pd

import streamlit as st

st.set_page_config(
    page_title="RaceMind",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ RaceMind")

st.subheader(
    "Formula 1 Strategy Intelligence Platform"
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

st.write(
    f"Selected Driver: {selected_driver}"
)

pit_lap = st.slider(
    "Pit Stop Lap",
    min_value=18,
    max_value=28,
    value=23
)

st.write(
    f"Selected Pit Lap: {pit_lap}"
)

st.divider()

st.subheader("Strategy Prediction")

medium_model = {
    "Intercept": 89.77,
    "Slope": -0.005
}

hard_model = {
    "Intercept": 87.93,
    "Slope": 0.0086
}

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

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicted Race Time",
        f"{predicted_time:.2f} sec"
    )

with col2:
    st.metric(
        "Recommended Pit Lap",
        str(int(best_strategy["PitLap"]))
    )

import plotly.express as px

fig = px.line(
    simulation_df,
    x="PitLap",
    y="PredictedRaceTime",
    markers=True,
    title="Strategy Simulation"
)

st.plotly_chart(
    fig,
    width="stretch"
)