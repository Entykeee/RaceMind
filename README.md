# RaceMind

RaceMind is a Formula 1 strategy intelligence platform designed to analyze tyre degradation, simulate pit stop strategies, and provide data-driven race recommendations through an interactive dashboard.

Using historical race data from FastF1, the platform models tyre performance across stints, evaluates multiple pit stop scenarios, and identifies the optimal strategy for selected drivers and races.

## Key Features

* Pit stop strategy simulation
* Optimal pit window recommendation
* Tyre degradation modelling using linear regression
* Interactive race strategy dashboard
* Driver-to-driver strategy comparison
* Predicted race outcome analysis
* AI-generated strategy insights

## Technology Stack

* Python
* Streamlit
* FastF1
* Pandas
* NumPy
* Plotly

## Methodology

### Data Acquisition

Race telemetry and lap-level data are collected using FastF1, including:

* Lap times
* Tyre compounds
* Tyre life
* Driver performance metrics

### Tyre Degradation Modelling

A regression-based degradation model is trained for each tyre compound to estimate lap-time evolution throughout a stint.

### Strategy Simulation

The platform evaluates multiple pit stop windows and predicts total race time for each strategy configuration.

### Performance Comparison

Drivers are compared based on:

* Optimal pit stop lap
* Predicted race time
* Relative performance delta

## Dashboard Components

### Strategy Simulation

Visualizes predicted race time across different pit stop windows and highlights the optimal strategy.

### AI Strategy Engineer

Generates strategy recommendations based on simulation outcomes and degradation trends.

### Tyre Degradation Analysis

Displays degradation behaviour across tyre compounds using regression-based modelling.

### Driver Comparison

Compares predicted strategy performance across multiple drivers.

### Winner Prediction

Identifies the strongest strategic contender for the selected Grand Prix.

## Future Enhancements

* Safety Car and Virtual Safety Car simulations
* Weather-aware strategy modelling
* Multi-stop race strategy optimization
* Machine Learning-based race outcome prediction
* Real-time race strategy recommendations
