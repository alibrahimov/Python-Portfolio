# Cross-Sectional Machine Learning Factor Strategy

## Overview
This project implements a production-grade, market-neutral quantitative trading strategy. It utilizes a Random Forest regressor to predict 21-day forward returns for a universe of 50 highly liquid US equities. By ranking these predictions, the strategy constructs a cash-neutral portfolio by going long the top decile and short the bottom decile.

## Alpha Methodology & Feature Engineering
Unlike traditional time-series forecasting, this model predicts the *relative cross-sectional performance* of assets. 

1. **Factor Generation:** * **Momentum:** 21-day percentage change.
   * **Volatility:** 21-day rolling standard deviation of log returns.
   * **Mean Reversion:** Negative 5-day rolling sum of log returns.
2. **Cross-Sectional Neutralization:** To remove the effect of the broader market regime (Market Beta), all daily factor exposures are neutralized using a cross-sectional Z-score: `(x - mean) / std`.
3. **Target Variable:** 21-day forward log returns.

## Model Architecture
* **Algorithm:** `RandomForestRegressor` from `scikit-learn` (200 estimators, max depth of 6 to prevent overfitting on noisy financial data).
* **Validation:** The dataset (2012-2025) is split temporally (70% Train, 30% Out-of-Sample Test) to prevent data leakage and look-ahead bias.

## Backtest & Performance Metrics
The backtester evaluates a daily rebalanced long/short decile portfolio. 

* **CAGR:** [Run the script and insert number here]%
* **Annualized Volatility:** [Run the script and insert number here]%
* **Sharpe Ratio:** [Run the script and insert number here]
* **Max Drawdown:** [Run the script and insert number here]%
* **Mean Information Coefficient (IC):** [Run the script and insert number here]

*(Note: Transaction costs, slippage, and borrow fees are currently excluded from this baseline model).*

## How to Run

1. Ensure dependencies are installed:
   ```bash
   pip install pandas numpy yfinance scikit-learn matplotlib
