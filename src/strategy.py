from src.degradation import (
    prepare_lap_times,
    get_stint_summary
)


def get_actual_strategy(driver_laps):

    summary = get_stint_summary(driver_laps)

    strategy = {
        "NumberOfStints": len(summary),
        "Compounds": summary["Compound"].tolist(),
        "LapsPerStint": summary["Laps"].tolist()
    }

    return strategy


def get_pit_lap(driver_laps):

    summary = get_stint_summary(driver_laps)

    first_stint_laps = summary.iloc[0]["Laps"]

    return int(first_stint_laps)


def estimate_race_time(driver_laps):

    clean_laps = prepare_lap_times(driver_laps)

    return clean_laps["LapTimeSeconds"].sum()

def simulate_pit_stop(driver_laps, pit_lap):

    actual_pit_lap = get_pit_lap(driver_laps)

    actual_race_time = estimate_race_time(driver_laps)

    lap_difference = actual_pit_lap - pit_lap

    estimated_gain = lap_difference * 0.15

    simulated_race_time = (
        actual_race_time - estimated_gain
    )

    return {
        "ActualPitLap": actual_pit_lap,
        "SimulatedPitLap": pit_lap,
        "ActualRaceTime": round(actual_race_time, 2),
        "SimulatedRaceTime": round(simulated_race_time, 2),
        "Difference": round(
            actual_race_time - simulated_race_time,
            2
        )
    }

def predict_hard_lap_time(
    tyre_life,
    intercept,
    slope
):

    return intercept + (
        slope * tyre_life
    )

def predict_hard_stint_time(
    laps,
    intercept,
    slope
):

    total_time = 0

    for tyre_life in range(1, laps + 1):

        lap_time = predict_hard_lap_time(
            tyre_life,
            intercept,
            slope
        )

        total_time += lap_time

    return total_time

def predict_strategy_time(
    medium_laps,
    hard_laps,
    medium_intercept,
    medium_slope,
    hard_intercept,
    hard_slope
):

    medium_time = 0

    for tyre_life in range(1, medium_laps + 1):

        lap_time = (
            medium_intercept
            + medium_slope * tyre_life
        )

        medium_time += lap_time

    hard_time = predict_hard_stint_time(
        hard_laps,
        hard_intercept,
        hard_slope
    )

    return medium_time + hard_time

def simulate_strategy_window(
    start_lap,
    end_lap,
    medium_model,
    hard_model
):

    results = []

    for pit_lap in range(
        start_lap,
        end_lap + 1
    ):

        race_time = predict_strategy_time(
            medium_laps=pit_lap,
            hard_laps=57 - pit_lap,
            medium_intercept=medium_model["Intercept"],
            medium_slope=medium_model["Slope"],
            hard_intercept=hard_model["Intercept"],
            hard_slope=hard_model["Slope"]
        )

        results.append({
            "PitLap": pit_lap,
            "PredictedRaceTime": race_time
        })

    return results

def find_optimal_pit_stop(simulation_df):

    return simulation_df.loc[
        simulation_df["PredictedRaceTime"].idxmin()
    ]