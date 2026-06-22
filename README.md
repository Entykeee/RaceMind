# RaceMind

**Formula 1 Strategy Intelligence Platform**

Predictive analytics platform for tyre degradation modelling, pit-stop optimization, and Formula 1 race strategy simulation using historical telemetry data.

---

## Features

* **Tyre Degradation Modelling** — IQR-filtered linear regression per compound with fuel-corrected lap time analysis
* **1-Stop Strategy Simulation** — Closed-form race-time calculation across the full pit window
* **2-Stop Strategy Analysis** — Comparative simulation with recommended strategy and projected time gain
* **Undercut Assessment** — Evaluates whether a fresh-tyre undercut can overcome the gap to the car ahead
* **Actual vs Optimal Strategy Comparison** — Benchmarks real pit-stop decisions against model recommendations
* **Prediction Validation** — Compares predicted race outcomes against historical race results
* **Interactive Dashboard** — Premium Formula 1-inspired analytics interface built with Streamlit and Plotly

---

## Highlights

* Fuel-corrected degradation modelling
* Historical Formula 1 telemetry analysis using FastF1
* Supports both 1-stop and 2-stop strategy evaluation
* Driver ranking based on simulated race outcomes
* Interactive visual analytics dashboard

---

## Tech Stack

| Layer         | Tools                                   |
| ------------- | --------------------------------------- |
| Data          | FastF1, Pandas                          |
| Modelling     | NumPy, Linear Regression, IQR Filtering |
| Visualization | Plotly                                  |
| Dashboard     | Streamlit                               |
| Styling       | Custom CSS                              |

---

## Project Structure

```text
RaceMind/
├── dashboard/
│   └── app.py
├── src/
│   ├── data_loader.py
│   ├── degradation.py
│   ├── strategy.py
│   ├── prediction.py
│   ├── ranking.py
│   └── validation.py
├── style.css
└── README.md
```

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/Entykeee/RaceMind.git
cd RaceMind
```

### Install Dependencies

```bash
pip install fastf1 streamlit plotly pandas numpy scipy
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

> First launch may take 15–30 seconds while FastF1 downloads and caches race data.

---

## How It Works

### 1. Data Collection

Race sessions and lap-level telemetry data are fetched through the FastF1 API.

### 2. Data Cleaning

Lap times are cleaned using IQR-based outlier filtering to reduce the influence of Safety Car laps, Virtual Safety Car laps, and other anomalous race events.

### 3. Fuel Correction

Fuel load naturally decreases throughout a race, making later laps faster.

RaceMind applies fuel correction before fitting degradation models to isolate tyre wear effects from fuel-burn effects.

### 4. Tyre Degradation Modelling

For each compound:

```text
Lap Time = Intercept + (Slope × Tyre Life)
```

The degradation slope represents the estimated pace loss per lap due to tyre wear.

### 5. Strategy Simulation

RaceMind evaluates every candidate pit lap and predicts total race time using degradation models.

```text
Race Time
=
Stint 1 Time
+ Pit Stop Delta
+ Stint 2 Time
```

Both 1-stop and 2-stop strategies can be analyzed and compared.

### 6. Recommendation Engine

The platform generates:

* Optimal pit windows
* Strategy recommendations
* Undercut assessments
* Driver rankings
* Actual vs optimal strategy comparisons

---

## Screenshots

### Strategy Dashboard

*Add screenshot*

### Pit Window Simulation

*Add screenshot*

### Tyre Degradation Analysis

*Add screenshot*

### Driver Ranking

*Add screenshot*

---

## Limitations

* Uses linear degradation assumptions and does not model tyre performance cliffs
* Does not account for Safety Car or Virtual Safety Car probability
* Does not model weather changes or track evolution
* Does not incorporate traffic or dirty-air effects
* Designed as a predictive analytics platform rather than a full Formula 1 race engineering simulator

---

## Future Improvements

* Tyre cliff detection
* Track evolution modelling
* Traffic-aware strategy simulation
* Historical backtesting framework
* Confidence intervals for strategy recommendations
* Safety Car scenario analysis

---

## Data Source

Race data is sourced through the FastF1 Python library, which provides access to official Formula 1 timing and telemetry feeds.

This project is not affiliated with Formula 1, the FIA, or any Formula 1 team.

---

## Author

**Ojas Godambe**

Built as a sports analytics and predictive modelling project inspired by Formula 1 race strategy engineering.
