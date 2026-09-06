# tk-ovr
lets see if this works.         10:12

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

## Status

**In active development.**

Future work includes ensemble models, alternative data, NLP-based information extraction, portfolio construction, and automated research reporting.

## Author

**Collin McDevitt**

Independent quantitative finance and machine-learning research project.
