# AR-GARCH Volatility-Targeted Trading Strategy

## Overview
This quantitative trading pipeline implements a time-series momentum strategy on the SPY ETF. It utilizes an Autoregressive model to generate directional signals and a GARCH model to dynamically scale position sizing based on forecasted market volatility.

## Alpha Methodology & Risk Management
1. **Directional Signal (AR Model):** A rolling 3-year window of log returns is used to fit an `ARIMA(1,0,0)` model. The sign of the 21-day forward forecast dictates whether the strategy takes a long or short position.
2. **Dynamic Volatility Targeting (GARCH Model):** Financial time series exhibit volatility clustering. To maintain a constant risk profile, the strategy fits a `GARCH(1,1)` model to forecast the upcoming month's variance. 
3. **Position Sizing:** The nominal exposure is scaled inversely to the forecasted volatility using the formula: `Target Volatility (10%) / Forecasted Volatility`. If the market is predicted to be highly volatile, leverage is reduced.

## Backtest Performance (2010 - 2025)
* **CAGR:** [Insert Number]%
* **Annualized Volatility:** [Insert Number]%
* **Sharpe Ratio:** [Insert Number]
* **Max Drawdown:** [Insert Number]%

## How to Run
Ensure dependencies are installed:
```bash
pip install pandas numpy yfinance statsmodels arch matplotlib
