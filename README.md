# tk-ovr

**Quantitative prediction-market alpha engine for detecting and evaluating potential market mispricing.**

tk-ovr analyzes prediction-market data to estimate the probability of future events, compare model probabilities against market-implied probabilities, and identify potential sources of alpha.

## What It Does

* Ingests prediction-market data through APIs
* Engineers market and time-series features
* Estimates independent event probabilities using statistical/ML models
* Compares model probabilities against market prices
* Generates potential alpha signals based on estimated edge
* Backtests signals using historical data
* Evaluates performance, calibration, and risk
* Visualizes opportunities and model performance through a research dashboard

## Core Pipeline

```text
Market Data
    ↓
Feature Engineering
    ↓
Probability Model
    ↓
Market vs. Model
    ↓
Alpha Signal
    ↓
Risk Management
    ↓
Backtesting
```

## Tech Stack

**Python · SQL · PostgreSQL · scikit-learn · APIs · FastAPI · React · Next.js · Docker**

## Research Focus

The central question behind tk-ovr is:

> **Can independently estimated probabilities identify persistent differences between prediction-market prices and fair value after accounting for risk, liquidity, and execution costs?**

The project emphasizes out-of-sample testing, probability calibration, realistic backtesting, and reproducible quantitative research.

## Progress

- [x] **Milestone 1 — Data ingestion.** `src/data/polymarket_api.py` pulls markets from Polymarket's Gamma API (cursor-based pagination), normalizes prices/volume/liquidity/resolution-date/status, and persists them as timestamped JSON snapshots and into PostgreSQL (`markets` + `market_snapshots` tables). Currently pulling ~10,000 active, liquid, not-yet-resolved markets per run, filtered and sorted by volume.
- [x] **Milestone 2 — Automated/scheduled ingestion.** A GitHub Actions workflow (`.github/workflows/scheduled_ingestion.yml`) runs the ingestion pipeline every hour automatically, batch-writing to Postgres via `execute_values`. `market_snapshots` now accumulates a real historical time series with no manual intervention.
- [ ] **Milestone 3 — Feature engineering.** Price momentum, rolling volatility, liquidity/volume changes, time-to-resolution features.
- [ ] **Milestone 4 — Probability models.** Market baseline → historical base rate → logistic regression → gradient boosting → ensemble.
- [ ] **Milestone 5 — Edge / expected value / calibration.**
- [ ] **Milestone 6 — Walk-forward backtesting.**
- [ ] **Milestone 7 — Research dashboard.**

## Status

**In active development.**

Future work includes ensemble models, alternative data, NLP-based information extraction, portfolio construction, and automated research reporting.

## Author

**Collin McDevitt**

Independent quantitative finance and machine-learning project.
