"""
data_loader.py
──────────────
Handles all FastF1 session and lap data fetching.
Centralises loading so the rest of the codebase never
touches the FastF1 API directly.
"""

import fastf1

import os
os.makedirs("./f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("./f1_cache")

def load_race_session(
    year: int = 2025,
    grand_prix: str = "Abu Dhabi",
    session_type: str = "R"
) -> fastf1.core.Session:
    """
    Load and return a FastF1 race session.

    Parameters
    ----------
    year         : Championship season (default 2025).
    grand_prix   : Event name as it appears in the schedule.
    session_type : 'R' = Race, 'Q' = Qualifying, etc.

    Returns
    -------
    fastf1.core.Session – fully loaded session object.
    """

    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    return session


def get_driver_laps(
    session: fastf1.core.Session,
    driver: str
):
    """
    Return all laps recorded for a single driver in a session.

    Parameters
    ----------
    session : Loaded FastF1 session.
    driver  : Three-letter driver code, e.g. 'VER'.

    Returns
    -------
    fastf1.core.Laps – lap data for the requested driver.
    """

    return session.laps.pick_drivers(driver)


def get_race_laps(session: fastf1.core.Session) -> int:
    """
    Return the scheduled race distance in laps.

    Falls back to the actual maximum lap number recorded if
    the session object does not expose `total_laps` directly
    (older FastF1 versions).

    Parameters
    ----------
    session : Loaded FastF1 session.

    Returns
    -------
    int – number of laps in the race.
    """

    if hasattr(session, "total_laps") and session.total_laps:
        return int(session.total_laps)

    if hasattr(session, "event") and "RaceLaps" in session.event:
        laps = session.event["RaceLaps"]
        if laps and laps > 0:
            return int(laps)

    # Fallback: derive from lap data
    return int(session.laps["LapNumber"].max())