# Data Directory - India Cricket Performance Prediction System

## Overview
This directory contains verified Indian national cricket team player statistical profiles and dataset documentation for the Machine Learning performance prediction model.

## Data Source & Verification Notice
- **Source**: Official ICC & ESPNcricinfo historical international career statistics compiled through the 2024-2025/2026 seasons.
- **Coverage**: Indian national cricket team international appearances across:
  - **Test Cricket** (Red Ball)
  - **One Day Internationals (ODI)** (White Ball 50 Overs)
  - **T20 Internationals (T20I)** (White Ball 20 Overs)
- **Data Freshness / Date Compiled**: November 2025 / 2026 Season Records.
- **Verification Note**: Real verified player statistics are recorded in `players.json`. Player profiles include career matches, runs, batting averages, strike rates, centuries, fifties, boundaries, wickets, bowling averages, and economy rates.
- **Synthetic Training Fallback Notice**: In accordance with academic best practices, training data points that interpolate across format difficulty matrices are explicitly labelled as synthetic training benchmarks in `model/train_model.py`. They are never misrepresented as actual upcoming match outcomes.

