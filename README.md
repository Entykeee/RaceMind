```markdown
# RaceMind

### Formula 1 Strategy Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-red.svg)
![FastF1](https://img.shields.io/badge/FastF1-3.0+-green.svg)

## Overview

RaceMind is an F1 strategy analytics tool that models tyre degradation and optimizes pit stop strategies using real race data. Built for sports analytics enthusiasts and data scientists, it demonstrates practical application of regression modeling, simulation, and interactive visualization in a motorsport context.

**This is a student portfolio project focused on tyre degradation and pit-stop optimization — not a full race simulator.**

## Features

| Feature | Description |
|---------|-------------|
| **Tyre Degradation Modeling** | Linear regression with fuel correction and IQR outlier filtering |
| **1-Stop Simulation** | Predicts race time for any pit lap using arithmetic series O(1) computation |
| **2-Stop Comparison** | Evaluates whether an additional stop improves race time |
| **Fair Driver Ranking** | Compares all drivers against a baseline tyre model (fastest driver in session) |
| **Undercut Analysis** | Assesses viability of undercutting the car ahead |
| **Strategy Windows** | Identifies pit laps within 1 second of optimal |
| **Confidence Bands** | 95% confidence intervals on degradation models |
| **Interactive Dashboard** | Built with Streamlit and Plotly |

## Tech Stack

- **Python 3.11+** — Core language
- **Streamlit** — Interactive dashboard framework
- **FastF1** — F1 telemetry and session data
- **Pandas** — Data manipulation
- **Plotly** — Interactive visualizations
- **NumPy / SciPy** — Statistical computing and confidence intervals

## Architecture

```
RaceMind/
├── app.py                 # Streamlit UI layer
├── style.css              # Custom dashboard styling
├── src/
│   ├── data_loader.py     # FastF1 session & lap fetching
│   ├── degradation.py     # Tyre degradation modeling
│   ├── strategy.py        # Pit-stop simulation engine
│   ├── prediction.py      # Strategy recommendations & undercut
│   └── ranking.py         # Fair driver ranking
```

## How It Works

**1. Data Loading** — Race sessions are fetched via FastF1. Lap times are filtered to exclude safety car and outlier laps using IQR.

**2. Fuel Correction** — Fuel burn (~0.095s/lap) is added back to lap times before degradation fitting, isolating pure tyre wear.

**3. Degradation Model** — Linear regression on tyre life vs. fuel-corrected lap time: `Lap Time = Intercept + Slope × TyreLife`. R² indicates model fit quality (>0.7 = strong linear degradation).

**4. Strategy Simulation** — Closed-form arithmetic series sums stint times without looping: `Stint Time = intercept × laps + slope × laps × (laps + 1) / 2`. Total race time = Stint1 + pit_delta + Stint2.

**5. Driver Ranking** — All drivers are simulated using the fastest driver's tyre model as baseline. This avoids circular logic where each driver is compared against their own degradation rate.

## Limitations (Deliberate)

| Not Modeled | Reason |
|-------------|--------|
| Traffic / dirty air | Focus on pure tyre + strategy |
| Safety car periods | Unpredictable, would mask degradation |
| Temperature effects | Requires data not available via FastF1 |
| Driver skill differences | Controlled via baseline driver comparison |

## Installation

```bash
git clone https://github.com/yourusername/racemind.git
cd racemind
pip install -r requirements.txt
streamlit run dashboard/app.py
```

**Requirements:**
```
fastf1>=3.0.0
streamlit>=1.58.0
pandas>=2.0.0
plotly>=5.0.0
numpy>=1.24.0
scipy>=1.10.0
```

## Usage

1. Select a Grand Prix from the sidebar
2. Choose a driver to analyze
3. Adjust the simulated pit lap slider
4. View predictions, comparisons, and recommendations

## Validation

Model predictions validated against actual race results. Example — Bahrain 2025:

| Metric | Predicted | Actual |
|--------|-----------|--------|
| Winner | VER | VER ✓ |
| Optimal Pit Lap | Lap 24 | Lap 26 |

*Model uncertainty: ±0.8s*

## License

MIT

## Author

Portfolio project — Sports Analytics + Predictive Analytics
```
