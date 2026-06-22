"""
degradation.py
──────────────
Tyre degradation modelling using linear regression on
cleaned, outlier-filtered lap time data.
"""

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

FUEL_START_KG   = 110.0   # FIA 2025 max fuel load (kg)
TIME_PER_KG     = 0.03    # seconds lost per kg of fuel (FIA-documented)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_seconds(lap_series) -> pd.Series:
    return lap_series.dt.total_seconds()


def _iqr_filter(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series.between(q1 - k * iqr, q3 + k * iqr)


def _fuel_correction(lap_number: pd.Series, race_laps: int) -> pd.Series:
    """
    Fuel correction per lap using FIA-documented figures.
    As fuel burns off, car gets lighter → faster.
    We ADD correction to normalise all laps to a zero-fuel baseline.

    correction = fuel_remaining_at_lap * time_per_kg
    fuel_remaining = ((race_laps - lap_number) / race_laps) * FUEL_START_KG
    """
    fuel_remaining = ((race_laps - lap_number) / race_laps) * FUEL_START_KG
    return fuel_remaining * TIME_PER_KG


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_lap_times(driver_laps, race_laps: int = 57) -> pd.DataFrame:
    """
    Convert lap times to seconds, apply FIA-based fuel correction, filter quick laps.
    """
    clean = driver_laps.pick_quicklaps().copy()
    clean["LapTimeSeconds"] = _to_seconds(clean["LapTime"])
    clean["LapTimeSeconds_FuelCorrected"] = (
        clean["LapTimeSeconds"] + _fuel_correction(clean["LapNumber"], race_laps)
    )
    clean = clean.dropna(subset=["TyreLife", "LapTimeSeconds_FuelCorrected"])
    return clean


def get_compound_degradation(
    driver_laps,
    compound: str,
    race_laps: int = 57
) -> dict | None:
    """
    Fit linear degradation model with FIA-based fuel correction.
    """
    clean = prepare_lap_times(driver_laps, race_laps)
    compound_laps = clean[clean["Compound"] == compound].copy()

    if len(compound_laps) < 3:
        return None

    mask = _iqr_filter(compound_laps["LapTimeSeconds_FuelCorrected"])
    compound_laps = compound_laps[mask]

    if len(compound_laps) < 3:
        return None

    x = compound_laps["TyreLife"].values.astype(float)
    y = compound_laps["LapTimeSeconds_FuelCorrected"].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)

    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "Compound":  compound,
        "Slope":     float(slope),
        "Intercept": float(intercept),
        "R2":        round(r2, 4),
        "LapCount":  int(len(compound_laps)),
    }


def get_degradation_score(driver_laps, race_laps: int = 57) -> float:
    clean = prepare_lap_times(driver_laps, race_laps)
    compounds = clean["Compound"].dropna().unique()
    slopes = []
    for compound in compounds:
        model = get_compound_degradation(driver_laps, compound, race_laps)
        if model:
            slopes.append(model["Slope"])
    return float(np.mean(slopes)) if slopes else 0.0


def get_stint_summary(driver_laps, race_laps: int = 57) -> pd.DataFrame:
    clean = prepare_lap_times(driver_laps, race_laps)
    rows = []
    for stint_id in sorted(clean["Stint"].dropna().unique()):
        stint_data = clean[clean["Stint"] == stint_id]
        if stint_data.empty:
            continue
        compound  = stint_data["Compound"].iloc[0]
        lap_times = stint_data["LapTimeSeconds"]
        if len(stint_data) >= 2:
            sl, _ = np.polyfit(
                stint_data["TyreLife"].values.astype(float),
                lap_times.values.astype(float), 1
            )
        else:
            sl = 0.0
        rows.append({
            "Stint":            int(stint_id),
            "Compound":         compound,
            "Laps":             len(stint_data),
            "AveragePace":      round(lap_times.mean(), 3),
            "DegradationSlope": round(float(sl), 4)
        })
    return pd.DataFrame(rows)


def get_compound_pace(driver_laps, race_laps: int = 57) -> pd.DataFrame:
    clean = prepare_lap_times(driver_laps, race_laps)
    return (
        clean.groupby("Compound")["LapTimeSeconds"]
        .mean().reset_index()
        .rename(columns={"LapTimeSeconds": "MeanLapTime"})
    )


def get_driver_strategy_summary(session_laps, race_laps: int = 57) -> pd.DataFrame:
    all_summaries = []
    for driver in session_laps["Driver"].dropna().unique():
        driver_laps = session_laps.pick_drivers(driver)
        summary = get_stint_summary(driver_laps, race_laps)
        summary.insert(0, "Driver", driver)
        all_summaries.append(summary)
    if not all_summaries:
        return pd.DataFrame()
    return pd.concat(all_summaries, ignore_index=True)