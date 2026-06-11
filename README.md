# 🏎️ RaceMind

**Formula 1 Strategy Intelligence Platform**

RaceMind is a Formula 1 analytics platform that models tyre degradation, simulates pit-stop strategies, evaluates undercut opportunities, and generates data-driven race recommendations using historical race telemetry from FastF1.

The project combines sports analytics, predictive modeling, and interactive visualization to explore how tyre performance influences race strategy and pit-stop timing.

---

## Overview

Formula 1 strategy decisions are often determined by tyre degradation, pit-stop timing, and stint management.

RaceMind provides a framework for:

* Modeling tyre degradation from historical race data
* Simulating pit-stop strategies
* Identifying optimal pit windows
* Comparing actual race strategies against model recommendations
* Evaluating undercut opportunities
* Ranking drivers using strategy simulations
* Validating predictions against real race outcomes

The goal is not to replicate the proprietary systems used by Formula 1 teams, but to demonstrate practical applications of predictive analytics and data-driven decision-making in motorsport.

---

## Features

### Tyre Degradation Modeling

* Fuel-corrected lap time analysis
* IQR-based outlier removal
* Compound-specific degradation models
* Linear regression–based pace prediction
* Stint-level degradation summaries

### Strategy Simulation

* 1-stop strategy simulation engine
* 2-stop strategy comparison
* Optimal pit window identification
* Predicted race time estimation
* Strategy recommendation engine

### Race Intelligence

* Actual vs optimal strategy comparison
* Undercut viability analysis
* Driver ranking system
* Strategy performance assessment

### Validation

* Prediction validation against completed races
* Predicted winner vs actual winner comparison
* Predicted pit lap vs actual pit lap comparison
* Error tracking for model evaluation

### Interactive Dashboard

* Premium Formula 1 inspired interface
* Interactive race and driver selection
* Dynamic strategy visualizations
* Degradation analysis charts
* Pit-stop optimization dashboard
* Driver comparison views

---

## Tech Stack

### Analytics

* Python
* Pandas
* NumPy
* SciPy

### Formula 1 Data

* FastF1

### Visualization

* Plotly
* Streamlit

### Frontend Styling

* Custom CSS
* F1-inspired dashboard design

---

## Project Structure

```text
RaceMind/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── data_loader.py
│   ├── degradation.py
│   ├── strategy.py
│   ├── prediction.py
│   ├── ranking.py
│   └── validation.py
│
├── style.css
│
└── README.md
```

---

## Core Analytics Workflow

### 1. Data Collection

Race data is retrieved from FastF1.

```text
FastF1 Session
      ↓
Driver Laps
      ↓
Tyre Stints
```

### 2. Data Cleaning

Race laps are filtered to remove:

* Safety Car laps
* Virtual Safety Car laps
* Extreme outliers

Fuel-load correction is then applied to reduce bias caused by fuel burn during the race.

### 3. Degradation Modeling

For each tyre compound:

```text
Lap Time
     vs
Tyre Life
```

A degradation model is fitted to estimate:

* Degradation rate (s/lap)
* Baseline pace
* Model quality (R²)

### 4. Strategy Simulation

For each pit-stop scenario:

```text
Stint 1
   +
Pit Stop Delta
   +
Stint 2
```

The simulator evaluates every pit window and identifies the strategy with the lowest predicted race time.

### 5. Recommendation Generation

RaceMind generates:

* Optimal pit lap
* Predicted race time
* Strategy recommendations
* Undercut assessments

---

## Example Use Cases

### Strategy Planning

* Determine the best lap to pit
* Compare early vs late pit windows
* Evaluate 1-stop vs 2-stop approaches

### Tyre Analysis

* Compare degradation across compounds
* Identify high-degradation stints
* Analyze tyre performance trends

### Driver Comparison

* Compare predicted race outcomes
* Evaluate strategy effectiveness
* Rank drivers using simulated scenarios

---

## Limitations

RaceMind is intentionally simplified compared to real Formula 1 strategy systems.

The current model does not explicitly account for:

* Safety Car probability
* Traffic and dirty air effects
* Weather changes
* Track evolution
* Competitor reaction modeling
* Team-specific race simulations

The platform is designed as a sports analytics and predictive modeling project rather than a full race engineering system.

---

## Future Improvements

Potential future work includes:

* Safety Car scenario simulation
* Tyre cliff detection
* Track evolution modeling
* Confidence intervals for predictions
* Historical strategy backtesting
* Enhanced validation framework
* Traffic-aware race simulations

---

## Key Learnings

This project demonstrates:

* Data cleaning and preprocessing
* Predictive analytics
* Regression modeling
* Simulation systems
* Sports analytics
* Interactive dashboard development
* Data storytelling and visualization

---

## Screenshots

Add screenshots of:

* Strategy Overview
* Pit Stop Simulation
* Tyre Degradation Analysis
* Driver Comparison Dashboard

---

## Author

Ojas

Built as a sports analytics and predictive modeling project inspired by Formula 1 race strategy engineering.
