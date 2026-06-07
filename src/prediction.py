"""
prediction.py
─────────────
High-level race outcome prediction layer.

Sits above strategy.py and degradation.py to produce
human-readable strategy verdicts, driver rankings, and
undercut / overcut assessments.

Nothing in this file calls FastF1 directly — it operates
only on pre-computed degradation models and simulation
DataFrames so it remains fast and fully testable.
"""

from __future__ import annotations

import pandas as pd

from src.strategy import (
    simulate_strategy_window,
    find_optimal_pit_stop,
    PIT_STOP_DELTA
)


# ── Lap-time prediction ───────────────────────────────────────────────────────

def predict_lap_time(
    tyre_life: int,
    intercept: float,
    slope: float
) -> float:
    """
    Predict a single lap time given tyre age and a linear model.

    Parameters
    ----------
    tyre_life : Lap number on the current set of tyres (1-indexed).
    intercept : Model intercept.
    slope     : Degradation slope (s/lap).

    Returns
    -------
    float – predicted lap time in seconds.
    """

    return intercept + slope * tyre_life


def predict_lap_time_series(
    laps: int,
    intercept: float,
    slope: float
) -> list[float]:
    """
    Predict lap times for an entire stint of `laps` laps.

    Returns
    -------
    list[float] – one predicted lap time per lap (index 0 = lap 1).
    """

    return [
        predict_lap_time(tyre_life, intercept, slope)
        for tyre_life in range(1, laps + 1)
    ]


# ── Strategy recommendation ───────────────────────────────────────────────────

def build_strategy_recommendation(
    driver: str,
    pit_lap: int,
    best_pit_lap: int,
    predicted_time: float,
    best_time: float,
    model_1: dict,
    model_2: dict
) -> str:
    """
    Build a context-aware, data-driven strategy recommendation
    string for display in the UI.

    Accounts for direction of difference (pit earlier vs later)
    and model quality (slope signs).

    Parameters
    ----------
    driver          : Driver code string.
    pit_lap         : Currently selected pit lap.
    best_pit_lap    : Model-optimal pit lap.
    predicted_time  : Predicted race time at pit_lap.
    best_time       : Predicted race time at best_pit_lap.
    model_1 / _2    : Degradation model dicts.

    Returns
    -------
    str – recommendation paragraph.
    """

    time_delta = abs(predicted_time - best_time)

    if pit_lap == best_pit_lap:
        return (
            f"Lap {pit_lap} is the optimal pit window for {driver}. "
            f"The degradation models for {model_1['Compound']} "
            f"and {model_2['Compound']} both converge on this lap "
            f"as the crossover point where the tyre performance delta "
            f"outweighs the track position cost of pitting."
        )

    direction = "earlier" if best_pit_lap < pit_lap else "later"
    stint_note = (
        "An earlier stop allows the second compound to run longer "
        "at its optimal pace window, reducing overall degradation cost."
        if direction == "earlier"
        else
        "A later stop maximises the first compound's pace before "
        "the performance cliff, reducing the second stint length "
        "on a potentially slower tyre."
    )

    compound_1 = model_1["Compound"].capitalize()
    compound_2 = model_2["Compound"].capitalize()
    deg_1      = model_1["Slope"]
    deg_2      = model_2["Slope"]

    return (
        f"The model identifies lap {best_pit_lap} as the optimal window "
        f"for {driver} — {abs(pit_lap - best_pit_lap)} lap(s) "
        f"{direction} than the current selection.\n\n"
        f"Running lap {pit_lap} instead carries a predicted cost of "
        f"~{time_delta:.2f}s. {stint_note}\n\n"
        f"{compound_1} degradation: {deg_1:+.3f}s/lap  |  "
        f"{compound_2} degradation: {deg_2:+.3f}s/lap"
    )


# ── Driver ranking ────────────────────────────────────────────────────────────

def rank_drivers_by_strategy(
    session,
    drivers: list[str],
    get_driver_laps_fn,
    get_compound_degradation_fn,
    race_laps: int,
    pit_window_min: int,
    pit_window_max: int
) -> pd.DataFrame:
    """
    For every driver, fit degradation models and simulate the full
    pit-stop window.  Return a ranked DataFrame.

    Parameters
    ----------
    session                   : Loaded FastF1 session.
    drivers                   : List of driver codes to evaluate.
    get_driver_laps_fn        : Callable(session, driver) → laps.
    get_compound_degradation_fn: Callable(laps, compound) → model.
    race_laps                 : Circuit lap count.
    pit_window_min/max        : Simulation search window.

    Returns
    -------
    pd.DataFrame sorted by PredictedTime ascending, with columns:
        Driver, BestPitLap, PredictedTime, Stint1Time,
        Stint2Time, Compound1, Compound2, Deg1, Deg2, Delta
    """

    rows = []

    for driver in drivers:

        laps = get_driver_laps_fn(session, driver)

        compounds = laps["Compound"].dropna().unique()
        if len(compounds) < 2:
            continue

        m1 = get_compound_degradation_fn(laps, compounds[0])
        m2 = get_compound_degradation_fn(laps, compounds[1])

        if m1 is None or m2 is None:
            continue

        results = simulate_strategy_window(
            pit_window_min, pit_window_max,
            m1, m2,
            race_laps=race_laps
        )

        if not results:
            continue

        sim_df = pd.DataFrame(results)
        best   = find_optimal_pit_stop(sim_df)

        rows.append({
            "Driver":        driver,
            "BestPitLap":    int(best["PitLap"]),
            "PredictedTime": round(best["PredictedRaceTime"], 2),
            "Stint1Time":    round(best["Stint1Time"], 2),
            "Stint2Time":    round(best["Stint2Time"], 2),
            "Compound1":     compounds[0],
            "Compound2":     compounds[1],
            "Deg1":          round(m1["Slope"], 4),
            "Deg2":          round(m2["Slope"], 4)
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("PredictedTime").reset_index(drop=True)
    df["Delta"] = (df["PredictedTime"] - df["PredictedTime"].iloc[0]).round(2)
    return df


# ── Undercut / overcut assessment ─────────────────────────────────────────────

def assess_undercut(
    gap_to_ahead: float,
    pit_lap: int,
    model_1: dict,
    model_2: dict,
    race_laps: int,
    undercut_laps: int = 3
) -> dict:
    """
    Assess whether undercutting the car ahead is viable given
    the current gap and degradation models.

    The undercut is viable when the net lap-time gain from
    fresh tyres over `undercut_laps` laps exceeds the gap.

    Parameters
    ----------
    gap_to_ahead  : Current gap to the car ahead in seconds.
    pit_lap       : Proposed pit lap.
    model_1 / _2  : Degradation models for each compound.
    race_laps     : Total race laps.
    undercut_laps : How many laps of overcut advantage to model.

    Returns
    -------
    dict with keys:
        Viable (bool), ProjectedGainPerLap (float),
        TotalProjectedGain (float), Verdict (str)
    """

    # Lap time on old tyres (age = pit_lap)
    old_tyre_laptime = (
        model_1["Intercept"] + model_1["Slope"] * pit_lap
    )

    # Lap time on fresh tyres (age = 1)
    fresh_laptime = model_2["Intercept"] + model_2["Slope"] * 1

    gain_per_lap      = old_tyre_laptime - fresh_laptime
    total_gain        = gain_per_lap * undercut_laps - PIT_STOP_DELTA
    viable            = total_gain > gap_to_ahead

    if viable:
        verdict = (
            f"Undercut viable. Fresh {model_2['Compound']} tyres "
            f"project ~{gain_per_lap:.2f}s/lap gain. "
            f"Net advantage over {undercut_laps} laps: "
            f"~{total_gain:.2f}s vs gap of {gap_to_ahead:.2f}s."
        )
    else:
        verdict = (
            f"Undercut marginal. Projected net gain "
            f"({total_gain:.2f}s) does not overcome "
            f"the current gap ({gap_to_ahead:.2f}s) within "
            f"{undercut_laps} laps."
        )

    return {
        "Viable":                viable,
        "ProjectedGainPerLap":   round(gain_per_lap, 3),
        "TotalProjectedGain":    round(total_gain, 3),
        "Verdict":               verdict
    }