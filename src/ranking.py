# src/ranking.py
"""
Fair driver ranking using baseline degradation models.
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
    baseline_driver: str = "VER"
) -> pd.DataFrame:
    """
    Rank drivers using a SINGLE baseline degradation model (e.g., Verstappen's tyres).
    
    This fixes the circular logic where each driver was compared against their own
    degradation rates. Now everyone runs the same tyre model.
    
    Parameters
    ----------
    baseline_driver : The driver whose tyre degradation becomes the standard.
                      Usually the fastest driver on track.
    """
    
    # Step 1: Get baseline degradation model from reference driver
    baseline_laps = get_driver_laps_fn(session, baseline_driver)
    baseline_compounds = baseline_laps["Compound"].dropna().unique()
    
    if len(baseline_compounds) < 2:
        return pd.DataFrame()  # Not enough data
    
    baseline_model_1 = get_compound_degradation_fn(baseline_laps, baseline_compounds[0])
    baseline_model_2 = get_compound_degradation_fn(baseline_laps, baseline_compounds[1])
    
    if baseline_model_1 is None or baseline_model_2 is None:
        return pd.DataFrame()
    
    # Step 2: Simulate EVERY driver using the SAME tyre models
    rows = []
    for driver in drivers:
        results = simulate_strategy_window(
            pit_window_min, pit_window_max,
            baseline_model_1, baseline_model_2,
            race_laps=race_laps
        )
        
        if not results:
            continue
            
        sim_df = pd.DataFrame(results)
        best = find_optimal_pit_stop(sim_df)
        
        rows.append({
            "Driver": driver,
            "PredictedTime": round(best["PredictedRaceTime"], 2),
            "OptimalPitLap": int(best["PitLap"]),
            "Compound1": baseline_model_1["Compound"],
            "Compound2": baseline_model_2["Compound"],
            "Deg1": round(baseline_model_1["Slope"], 4),
            "Deg2": round(baseline_model_2["Slope"], 4)
        })
    
    if not rows:
        return pd.DataFrame()
    
    # Step 3: Sort and add delta column
    df = pd.DataFrame(rows).sort_values("PredictedTime").reset_index(drop=True)
    df["DeltaToLeader"] = (df["PredictedTime"] - df["PredictedTime"].iloc[0]).round(2)
    
    return df