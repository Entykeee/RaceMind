"""
strategy.py
───────────
Pit-stop strategy simulation engine.

Core idea
─────────
For a 1-stop race, the total predicted time is the sum of
two stints modelled by linear degradation curves:

    T(pit_lap) = Σ(lap=1..pit_lap)  [I₁ + S₁·lap]
               + pit_stop_delta
               + Σ(lap=1..race_laps-pit_lap) [I₂ + S₂·lap]

The inner sums are arithmetic series, so each is evaluated
in O(1) rather than looping lap-by-lap.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.degradation import (
    get_stint_summary,
    prepare_lap_times
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Typical time cost of a stationary pit stop (tyre change + release).
# 2024/25 average across all circuits is ~22–24 s.
PIT_STOP_DELTA: float = 23.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _arithmetic_stint_time(laps: int, intercept: float, slope: float) -> float:
    """
    Sum of (intercept + slope·k) for k = 1..laps.

    Closed-form:  intercept·laps + slope·laps·(laps+1)/2

    Exact and O(1).  No loop needed.
    """
    return intercept * laps + slope * laps * (laps + 1) / 2


# ── Strategy analysis helpers ─────────────────────────────────────────────────

def get_actual_strategy(driver_laps) -> dict:
    """
    Extract the actual strategy used by a driver from lap data.

    Returns
    -------
    dict with keys:
        NumberOfStints, Compounds, LapsPerStint, PitLaps
    """

    summary = get_stint_summary(driver_laps)

    pit_laps = []
    cumulative = 0
    for laps in summary["Laps"].tolist()[:-1]:   # all stints except last
        cumulative += laps
        pit_laps.append(cumulative)

    return {
        "NumberOfStints": len(summary),
        "Compounds":      summary["Compound"].tolist(),
        "LapsPerStint":   summary["Laps"].tolist(),
        "PitLaps":        pit_laps
    }


def get_pit_lap(driver_laps) -> int:
    """
    Return the lap on which the driver made their first pit stop,
    derived from stint data.
    """

    summary = get_stint_summary(driver_laps)
    return int(summary.iloc[0]["Laps"])


def estimate_race_time(driver_laps) -> float:
    """
    Sum of all quick-lap times in seconds – a simple actual
    race-time estimate for comparison purposes.
    """

    clean = prepare_lap_times(driver_laps)
    return float(clean["LapTimeSeconds"].sum())


# ── Core simulation engine ────────────────────────────────────────────────────

def predict_stint_time(laps: int, intercept: float, slope: float) -> float:
    """
    Predicted total time for a stint of `laps` laps starting
    on fresh tyres, using a linear degradation model.

    Parameters
    ----------
    laps      : Number of laps in the stint.
    intercept : Model intercept (predicted lap time at tyre life = 0).
    slope     : Degradation rate (seconds added per lap of tyre age).

    Returns
    -------
    float – total stint time in seconds.
    """

    return _arithmetic_stint_time(laps, intercept, slope)


def predict_strategy_time(
    stint1_laps:  int,
    stint2_laps:  int,
    model_1:      dict,
    model_2:      dict,
    pit_delta:    float = PIT_STOP_DELTA
) -> float:
    """
    Total predicted race time for a 1-stop strategy.

    Parameters
    ----------
    stint1_laps : Laps on the first compound (= pit lap number).
    stint2_laps : Laps on the second compound (= race_laps - pit_lap).
    model_1     : Degradation model dict for compound 1.
    model_2     : Degradation model dict for compound 2.
    pit_delta   : Pit-stop time loss in seconds.

    Returns
    -------
    float – total race time in seconds.
    """

    t1 = predict_stint_time(
        stint1_laps,
        model_1["Intercept"],
        model_1["Slope"]
    )

    t2 = predict_stint_time(
        stint2_laps,
        model_2["Intercept"],
        model_2["Slope"]
    )

    return t1 + pit_delta + t2


def simulate_strategy_window(
    start_lap:  int,
    end_lap:    int,
    model_1:    dict,
    model_2:    dict,
    race_laps:  int = 57,
    pit_delta:  float = PIT_STOP_DELTA
) -> list[dict]:
    """
    Simulate every possible 1-stop pit lap in [start_lap, end_lap]
    and return the predicted race time for each.

    Parameters
    ----------
    start_lap : First pit lap to evaluate.
    end_lap   : Last pit lap to evaluate (inclusive).
    model_1   : Degradation model for the first-stint compound.
    model_2   : Degradation model for the second-stint compound.
    race_laps : Total race distance in laps (circuit-specific).
    pit_delta : Pit-stop time loss in seconds.

    Returns
    -------
    list of dicts with keys PitLap, PredictedRaceTime,
    Stint1Time, Stint2Time.
    """

    results = []

    for pit_lap in range(start_lap, end_lap + 1):

        stint2_laps = race_laps - pit_lap

        if stint2_laps <= 0:
            continue

        t1 = predict_stint_time(
            pit_lap,
            model_1["Intercept"],
            model_1["Slope"]
        )

        t2 = predict_stint_time(
            stint2_laps,
            model_2["Intercept"],
            model_2["Slope"]
        )

        results.append({
            "PitLap":            pit_lap,
            "PredictedRaceTime": round(t1 + pit_delta + t2, 3),
            "Stint1Time":        round(t1, 3),
            "Stint2Time":        round(t2, 3)
        })

    return results


def find_optimal_pit_stop(simulation_df: pd.DataFrame) -> pd.Series:
    """
    Return the row in simulation_df with the lowest
    PredictedRaceTime.
    """

    return simulation_df.loc[
        simulation_df["PredictedRaceTime"].idxmin()
    ]


def compare_actual_vs_optimal(
    driver_laps,
    model_1: dict,
    model_2: dict,
    race_laps: int,
    pit_window_min: int,
    pit_window_max: int
) -> dict:
    """
    Compare the driver's actual pit lap against the model-optimal
    pit lap and return a structured summary.

    Parameters
    ----------
    driver_laps     : fastf1.core.Laps for the driver.
    model_1 / _2    : Degradation models for each compound.
    race_laps       : Circuit lap count.
    pit_window_min/max : Search window bounds.

    Returns
    -------
    dict with keys:
        ActualPitLap, OptimalPitLap,
        ActualPredictedTime, OptimalPredictedTime,
        TimeDelta, Verdict
    """

    actual_pit = get_pit_lap(driver_laps)

    results = simulate_strategy_window(
        pit_window_min, pit_window_max,
        model_1, model_2,
        race_laps=race_laps
    )

    sim_df  = pd.DataFrame(results)
    optimal = find_optimal_pit_stop(sim_df)

    actual_row = sim_df[sim_df["PitLap"] == actual_pit]
    actual_time = (
        actual_row["PredictedRaceTime"].iloc[0]
        if not actual_row.empty
        else None
    )

    delta = (
        round(actual_time - optimal["PredictedRaceTime"], 3)
        if actual_time is not None else None
    )

    if delta is None:
        verdict = "Actual pit lap outside simulation window."
    elif abs(delta) < 1.0:
        verdict = "Actual strategy was near-optimal."
    elif delta > 0:
        verdict = (
            f"Pitting on lap {int(optimal['PitLap'])} would have "
            f"saved ~{delta:.2f}s."
        )
    else:
        verdict = (
            f"Actual pit lap {actual_pit} was better than the "
            f"model optimum by {abs(delta):.2f}s — driver timing "
            f"or track position likely influenced this."
        )

    return {
        "ActualPitLap":          actual_pit,
        "OptimalPitLap":         int(optimal["PitLap"]),
        "ActualPredictedTime":   actual_time,
        "OptimalPredictedTime":  round(optimal["PredictedRaceTime"], 3),
        "TimeDelta":             delta,
        "Verdict":               verdict
    }

# strategy.py - Add these functions

def predict_2stop_strategy_time(
    stint1_laps: int,   # laps on compound 1 (lap 1 to pit1)
    stint2_laps: int,   # laps on compound 2 (pit1+1 to pit2)
    stint3_laps: int,   # laps on compound 3 (pit2+1 to end)
    model_1: dict,
    model_2: dict,
    model_3: dict,
    pit_delta: float = PIT_STOP_DELTA
) -> float:
    """
    Total predicted race time for a 2-stop strategy (3 stints).
    """
    t1 = predict_stint_time(stint1_laps, model_1["Intercept"], model_1["Slope"])
    t2 = predict_stint_time(stint2_laps, model_2["Intercept"], model_2["Slope"])
    t3 = predict_stint_time(stint3_laps, model_3["Intercept"], model_3["Slope"])
    
    return t1 + pit_delta + t2 + pit_delta + t3


def simulate_2stop_window(
    start_lap1: int,
    end_lap1: int,
    start_lap2: int,
    end_lap2: int,
    model_1: dict,
    model_2: dict,
    model_3: dict,
    race_laps: int,
    pit_delta: float = PIT_STOP_DELTA
) -> list[dict]:
    """
    Simulate all possible 2-stop pit lap combinations.
    
    WARNING: This is O(n²) — use with reasonable windows (e.g., 15-40 laps).
    For race_laps=60, that's ~600 combinations → fine.
    """
    results = []
    
    for pit1 in range(start_lap1, min(end_lap1, race_laps - 10) + 1):
        for pit2 in range(max(pit1 + 5, start_lap2), min(end_lap2, race_laps - 5) + 1):
            
            stint1 = pit1
            stint2 = pit2 - pit1
            stint3 = race_laps - pit2
            
            if stint2 < 3 or stint3 < 3:  # Minimum 3 laps per stint
                continue
            
            total_time = predict_2stop_strategy_time(
                stint1, stint2, stint3,
                model_1, model_2, model_3,
                pit_delta
            )
            
            results.append({
                "PitLap1": pit1,
                "PitLap2": pit2,
                "Stint1Laps": stint1,
                "Stint2Laps": stint2,
                "Stint3Laps": stint3,
                "PredictedRaceTime": round(total_time, 3)
            })
    
    return results


def find_optimal_2stop(simulation_results: list[dict]) -> dict:
    """Return the best 2-stop strategy."""
    if not simulation_results:
        return {}
    return min(simulation_results, key=lambda x: x["PredictedRaceTime"])

def find_optimal_2stop(simulation_results: list[dict]) -> dict:
    """Return the best 2-stop strategy."""
    if not simulation_results:
        return {}
    return min(simulation_results, key=lambda x: x["PredictedRaceTime"])

def compare_1stop_vs_2stop(
    model_1: dict,
    model_2: dict,
    race_laps: int,
    pit_window_min: int = 15,
    pit_window_max: int = 40
) -> dict:
    """
    Compare optimal 1-stop vs optimal 2-stop strategy.
    Returns dict with both strategies and recommended approach.
    """
    from src.strategy import simulate_strategy_window, find_optimal_pit_stop
    
    # 1-stop simulation
    one_stop_results = simulate_strategy_window(
        pit_window_min, pit_window_max,
        model_1, model_2,
        race_laps
    )
    one_stop_best = find_optimal_pit_stop(pd.DataFrame(one_stop_results))
    
    # 2-stop simulation (reusing compounds: 1 → 2 → 1)
    two_stop_results = simulate_2stop_window(
        pit_window_min, pit_window_max - 10,
        pit_window_min + 5, pit_window_max,
        model_1, model_2, model_1,
        race_laps
    )
    two_stop_best = find_optimal_2stop(two_stop_results)
    
    if not two_stop_best:
        return {"Error": "Not enough laps for 2-stop simulation"}
    
    time_diff = one_stop_best["PredictedRaceTime"] - two_stop_best["PredictedRaceTime"]
    
    return {
        "OneStop": {
            "PitLap": int(one_stop_best["PitLap"]),
            "PredictedTime": one_stop_best["PredictedRaceTime"]
        },
        "TwoStop": {
            "PitLap1": two_stop_best["PitLap1"],
            "PitLap2": two_stop_best["PitLap2"],
            "PredictedTime": two_stop_best["PredictedRaceTime"]
        },
        "TimeDifference": round(abs(time_diff), 2),
        "Recommended": "2-stop" if time_diff > 0 else "1-stop",
        "GainFrom2Stop": round(time_diff, 2)
    }