"""
validation.py
Validates RaceMind predictions against actual race results.
"""

import pandas as pd
import fastf1

def validate_race_prediction(
    year: int,
    grand_prix: str,
    predicted_winner: str,
    predicted_pit_lap: int
) -> dict:
    """
    Compare model prediction against actual race result.
    """
    session = fastf1.get_session(year, grand_prix, 'R')
    session.load()
    
    # Get actual winner
    results = session.results
    actual_winner = results.iloc[0]['Abbreviation']
    
    # Get actual winner's pit lap (first stop)
    winner_laps = session.laps.pick_drivers(actual_winner)
    stints = winner_laps['Stint'].dropna().unique()
    actual_pit = None
    
    if len(stints) > 1:
        first_stint_laps = winner_laps[winner_laps['Stint'] == stints[0]]
        actual_pit = int(first_stint_laps['LapNumber'].max())
    
    return {
        "GrandPrix": grand_prix,
        "PredictedWinner": predicted_winner,
        "ActualWinner": actual_winner,
        "Correct": predicted_winner == actual_winner,
        "PredictedPitLap": predicted_pit_lap,
        "ActualPitLap": actual_pit,
        "PitLapError": abs(predicted_pit_lap - actual_pit) if actual_pit else None
    }


def run_validation_series(race_predictions: list[dict]) -> pd.DataFrame:
    """
    Run validation for multiple races.
    
    race_predictions = [
        {"year": 2025, "grand_prix": "Bahrain", "predicted_winner": "VER", "predicted_pit_lap": 18},
        ...
    ]
    """
    results = []
    for pred in race_predictions:
        results.append(validate_race_prediction(**pred))
    return pd.DataFrame(results)