"""
degradation.py
──────────────
Tyre degradation modelling using linear regression on
cleaned, outlier-filtered lap time data.

All public functions return plain dicts or DataFrames so
callers never need to import numpy or sklearn.
"""

import numpy as np
import pandas as pd

# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_seconds(lap_series) -> pd.Series:
    """Convert a timedelta Series to float seconds."""
    return lap_series.dt.total_seconds()


def _iqr_filter(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Return a boolean mask that is True for values inside the
    [Q1 - k·IQR, Q3 + k·IQR] fence.

    Removes safety-car laps, VSC laps, and slow-zone
    outliers that would otherwise distort the slope fit.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series.between(q1 - k * iqr, q3 + k * iqr)


# ── Public API ────────────────────────────────────────────────────────────────

FUEL_BURN_PER_LAP = 0.095  # seconds per lap (F1 2025 average)

def prepare_lap_times(driver_laps) -> pd.DataFrame:
    """
    Convert lap times to seconds, remove fuel effect, filter quick laps.
    """
    clean = driver_laps.pick_quicklaps().copy()
    clean["LapTimeSeconds"] = _to_seconds(clean["LapTime"])
    
    # Fuel correction: later laps are faster due to less fuel
    # We ADD fuel burn * lap number because older laps had MORE fuel
    # This gives us the "true" tyre degradation without fuel masking it
    clean["LapTimeSeconds_FuelCorrected"] = (
        clean["LapTimeSeconds"] + FUEL_BURN_PER_LAP * clean["LapNumber"]
    )
    
    clean = clean.dropna(subset=["TyreLife", "LapTimeSeconds_FuelCorrected"])
    return clean


def get_compound_degradation(
    driver_laps,
    compound: str,
    use_fuel_correction: bool = True
) -> dict | None:
    """
    Fit linear degradation model with optional fuel correction.
    """
    clean = prepare_lap_times(driver_laps)
    
    compound_laps = clean[clean["Compound"] == compound].copy()
    
    if len(compound_laps) < 3:
        return None
    
    # Outlier removal
    time_col = "LapTimeSeconds_FuelCorrected" if use_fuel_correction else "LapTimeSeconds"
    mask = _iqr_filter(compound_laps[time_col])
    compound_laps = compound_laps[mask]
    
    if len(compound_laps) < 3:
        return None
    
    x = compound_laps["TyreLife"].values.astype(float)
    y = compound_laps[time_col].values.astype(float)
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # R² calculation
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
        "FuelCorrected": use_fuel_correction
    }

def get_degradation_score(driver_laps) -> float:
    """
    Return a single overall degradation rate (seconds/lap)
    across all compounds combined.

    Uses get_compound_degradation internally to benefit from
    outlier filtering.

    Parameters
    ----------
    driver_laps : fastf1.core.Laps for a single driver.

    Returns
    -------
    float – seconds lost per lap (positive = getting slower).
    """

    clean = prepare_lap_times(driver_laps)
    compounds = clean["Compound"].dropna().unique()

    slopes = []
    for compound in compounds:
        model = get_compound_degradation(driver_laps, compound)
        if model:
            slopes.append(model["Slope"])

    return float(np.mean(slopes)) if slopes else 0.0


def get_stint_summary(driver_laps) -> pd.DataFrame:
    """
    Summarise each stint: compound used, lap count,
    average pace, and degradation slope.

    Parameters
    ----------
    driver_laps : fastf1.core.Laps for a single driver.

    Returns
    -------
    pd.DataFrame with columns:
        Stint, Compound, Laps, AveragePace, DegradationSlope
    """

    clean = prepare_lap_times(driver_laps)
    rows = []

    for stint_id in sorted(clean["Stint"].dropna().unique()):

        stint_data = clean[clean["Stint"] == stint_id]

        if stint_data.empty:
            continue

        compound  = stint_data["Compound"].iloc[0]
        lap_times = stint_data["LapTimeSeconds"]

        # Per-stint slope via polyfit (no cross-compound noise)
        if len(stint_data) >= 2:
            sl, _ = np.polyfit(
                stint_data["TyreLife"].values.astype(float),
                lap_times.values.astype(float),
                1
            )
        else:
            sl = 0.0

        rows.append({
            "Stint":             int(stint_id),
            "Compound":          compound,
            "Laps":              len(stint_data),
            "AveragePace":       round(lap_times.mean(), 3),
            "DegradationSlope":  round(float(sl), 4)
        })

    return pd.DataFrame(rows)


def get_compound_pace(driver_laps) -> pd.DataFrame:
    """
    Return mean lap time per compound for a driver.

    Parameters
    ----------
    driver_laps : fastf1.core.Laps for a single driver.

    Returns
    -------
    pd.DataFrame with columns Compound, LapTimeSeconds.
    """

    clean = prepare_lap_times(driver_laps)
    return (
        clean
        .groupby("Compound")["LapTimeSeconds"]
        .mean()
        .reset_index()
        .rename(columns={"LapTimeSeconds": "MeanLapTime"})
    )


def get_driver_strategy_summary(session_laps) -> pd.DataFrame:
    """
    Build a per-driver, per-stint summary for all drivers
    in a session.

    Parameters
    ----------
    session_laps : session.laps (all drivers).

    Returns
    -------
    pd.DataFrame with a 'Driver' column prepended.
    """

    all_summaries = []

    for driver in session_laps["Driver"].dropna().unique():
        driver_laps = session_laps.pick_drivers(driver)
        summary = get_stint_summary(driver_laps)
        summary.insert(0, "Driver", driver)
        all_summaries.append(summary)

    if not all_summaries:
        return pd.DataFrame()

    return pd.concat(all_summaries, ignore_index=True)