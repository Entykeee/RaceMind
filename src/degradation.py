import pandas as pd


def prepare_lap_times(driver_laps):
    """
    Convert lap times to seconds and keep only quick laps.
    """

    clean_laps = driver_laps.pick_quicklaps().copy()

    clean_laps["LapTimeSeconds"] = (
        clean_laps["LapTime"]
        .dt.total_seconds()
    )

    return clean_laps

import numpy as np


def get_degradation_score(driver_laps):
    """
    Calculate tyre degradation rate
    using a simple linear fit.

    Returns:
        seconds lost per lap
    """

    clean_laps = prepare_lap_times(driver_laps)

    x = clean_laps["TyreLife"]
    y = clean_laps["LapTimeSeconds"]

    slope, intercept = np.polyfit(x, y, 1)

    return slope

def get_stint_summary(driver_laps):

    clean_laps = prepare_lap_times(driver_laps)

    summary = []

    for stint in clean_laps['Stint'].unique():

        stint_data = clean_laps[
            clean_laps['Stint'] == stint
        ]

        summary.append({
            'Stint': int(stint),
            'Compound': stint_data['Compound'].iloc[0],
            'Laps': len(stint_data),
            'AveragePace': round(
                stint_data['LapTimeSeconds'].mean(),
                3
            )
        })

    return pd.DataFrame(summary)

def get_driver_strategy_summary(laps):

    drivers = laps['Driver'].unique()

    all_summaries = []

    for driver in drivers:

        driver_laps = laps.pick_drivers(driver)

        summary = get_stint_summary(driver_laps)

        summary['Driver'] = driver

        all_summaries.append(summary)

    return pd.concat(all_summaries, ignore_index=True)

def get_compound_pace(driver_laps):

    clean_laps = prepare_lap_times(driver_laps)

    summary = (
        clean_laps
        .groupby("Compound")["LapTimeSeconds"]
        .mean()
        .reset_index()
    )

    return summary

def get_compound_degradation(driver_laps, compound):

    clean_laps = prepare_lap_times(driver_laps)

    compound_laps = clean_laps[
        clean_laps["Compound"] == compound
    ]

    x = compound_laps["TyreLife"]
    y = compound_laps["LapTimeSeconds"]

    slope, intercept = np.polyfit(x, y, 1)

    return {
        "Compound": compound,
        "Slope": slope,
        "Intercept": intercept
    }