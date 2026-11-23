import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

tickers = ["KO", "PEP"]
start = "2010-01-01"
end = "2025-01-01"

prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"].dropna()

score, pvalue, _ = coint(prices[tickers[0]], prices[tickers[1]])

if pvalue > 0.05:
    raise ValueError("Pair is not cointegrated")

X = sm.add_constant(prices[tickers[1]])
model = sm.OLS(prices[tickers[0]], X).fit()
hedge_ratio = model.params[1]

spread = prices[tickers[0]] - hedge_ratio * prices[tickers[1]]
zscore = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

entry = 2.0
exit = 0.5

positions = pd.DataFrame(index=prices.index, columns=tickers)
positions[:] = 0

positions.loc[zscore > entry, tickers[0]] = -1
positions.loc[zscore > entry, tickers[1]] = hedge_ratio

positions.loc[zscore < -entry, tickers[0]] = 1
positions.loc[zscore < -entry, tickers[1]] = -hedge_ratio

positions[(zscore.abs() < exit)] = 0

positions = positions.shift().fillna(0)

returns = prices.pct_change().fillna(0)
strategy_returns = (positions * returns).sum(axis=1)

cum_returns = (1 + strategy_returns).cumprod()

cagr = cum_returns.iloc[-1] ** (252 / len(strategy_returns)) - 1
vol = strategy_returns.std() * np.sqrt(252)
sharpe = cagr / vol

drawdown = cum_returns / cum_returns.cummax() - 1
max_dd = drawdown.min()

results = pd.DataFrame({
    "Metric": ["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown", "Cointegration p-value"],
    "Value": [cagr, vol, sharpe, max_dd, pvalue]
})

print(results)
