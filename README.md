# Expected Goals (xG) Modeling and Player Valuation

This repository provides a pipeline for building Expected Goals (xG) models using Gaussian Processes and using them for football player valuation.

## Overview

The `main.py` script orchestrates an end-to-end pipeline:

1.  **Data Processing (`src.features.data_processing`):** Consolidates and preprocesses raw event and tracking data, normalizes coordinates, and filters for relevant shot events.
2.  **Feature Engineering (`src.features.feature_engineering`):** Generates features for xG prediction, splits data, and scales features.
3.  **Gaussian Proccesses Model Training (`src.models.gp_xg_models`):** Trains Variational Gaussian Process (VGP) and Sparse VGP (SVGP) classifiers to predict goal probability. Models and feature importance (ARD lengthscales) are saved. Other machine learning approaches are available in `src.models.xg_models` for comparative analysis.
4.  **Player Valuation (`src.models.value_players`):** Uses VGP model outputs (xG mean and variance) to calculate player performance metrics and soft-weighted values. `src.models.value_players` also includes a class to marginalizing over shot characteristics to estimate player-specific shooting ability.
5.  **Visualization & Results (`src.viz.plot_utils`):** Generates plots like xG distributions and pitch heatmaps.

Unfortunately due to privacy constraints the data cannot be provided. However, this process is demonstrated in the `src.notebooks.xg_models`

## Repository Structure

*   `main.py`: Main pipeline script.
*   `src/`: Core modules for data processing, feature engineering, GP models, player valuation, and plotting utilities.
*   `data/`: For raw, processed, and feature data.
*   `models/gp/`: Stores trained GP models.
*   `results/`: Outputs including player value CSVs and plots.
*   `logs/`: Pipeline execution logs.
*   `requirements.txt`: Project dependencies.

## How to Run

1.  **Setup:** Clone repo and `pip install -r requirements.txt`.
2.  **Data:** Place raw data in `data/raw/`. Cannot be shared for privacy reasons.
3.  **Execute:** `python main.py`.
4.  **Outputs:** Check `models/`, `results/`, and `logs/` directories.

## Key Dependencies

*   See `requirements.txt` for a full list.

## Future Ideas
*   Test marginalization with more data and incorporate uncertainty.
*   Assess other actions such as passing and defending.



