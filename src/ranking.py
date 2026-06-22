# src/ranking.py
"""
Driver ranking using each driver's own degradation model.
Gap is relative to the fastest predicted time on the grid.
"""

import pandas as pd
from src.strategy import simulate_strategy_window, find_optimal_pit_stop


def rank_drivers_fair(
    session,
    drivers: list[str],
    get_driver_laps_fn,
    get_compound_degradation_fn,
    race_laps: int,
    pit_window_min: int,
    pit_window_max: int,
    baseline_driver: str = None  # unused, kept for API compatibility
) -> pd.DataFrame:
    """
    Rank drivers using each driver's own degradation model.
    Gap is calculated relative to the fastest predicted time.
    """

    rows = []

    for driver in drivers:
        laps      = get_driver_laps_fn(session, driver)
        compounds = laps["Compound"].dropna().unique().tolist()

        if len(compounds) < 2:
            continue

        m1 = get_compound_degradation_fn(laps, compounds[0], race_laps)
        m2 = get_compound_degradation_fn(laps, compounds[1], race_laps)

        if m1 is None or m2 is None:
            continue

        results = simulate_strategy_window(
            pit_window_min, pit_window_max, m1, m2, race_laps=race_laps
        )

        if not results:
            continue

        sim_df = pd.DataFrame(results)
        best   = find_optimal_pit_stop(sim_df)

        rows.append({
            "Driver":        driver,
            "PredictedTime": round(best["PredictedRaceTime"], 2),
            "OptimalPitLap": int(best["PitLap"]),
            "Compound1":     m1["Compound"],
            "Compound2":     m2["Compound"],
            "Deg1":          round(m1["Slope"], 4),
            "Deg2":          round(m2["Slope"], 4),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("PredictedTime").reset_index(drop=True)
    df["DeltaToLeader"] = (df["PredictedTime"] - df["PredictedTime"].iloc[0]).round(2)

    return df