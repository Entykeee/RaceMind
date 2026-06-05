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