import yfinance as yf
import numpy as np
from scipy import stats
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
data = yf.download(tickers, start="2024-01-01", end="2025-01-01")["Adj Close"].pct_change().dropna()

alpha = 0.05
results = []

for t in tickers[:-1]:
    r = data[t]
    n = len(r)
    mean = np.mean(r)
    std = np.std(r, ddof=1)
    se = std / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(r, 0)
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    decision = "Reject H0" if p_val < alpha else "Fail to reject H0"
    results.append([t, mean, ci[0], ci[1], t_stat, p_val, decision])

res_ttest = pd.DataFrame(results, columns=["Ticker", "Mean", "CI Lower", "CI Upper", "t", "p", "Decision"])

corr_results = []
spy = data["SPY"]
for t in tickers[:-1]:
    corr, _ = stats.pearsonr(data[t], spy)
    fisher_z = np.arctanh(corr)
    se = 1 / np.sqrt(len(data[t]) - 3)
    z_stat = fisher_z / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    decision = "Reject H0" if p_val < alpha else "Fail to reject H0"
    corr_results.append([t, corr, z_stat, p_val, decision])

res_corr = pd.DataFrame(corr_results, columns=["Ticker", "Corr with SPY", "z", "p", "Decision"])

print("One-sample t-tests (H0: μ=0):")
print(res_ttest)
print("\nCorrelation tests (H0: corr=0):")
print(res_corr)
