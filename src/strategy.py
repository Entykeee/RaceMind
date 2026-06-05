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