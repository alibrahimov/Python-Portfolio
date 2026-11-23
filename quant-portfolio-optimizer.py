import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
data = yf.download(tickers, start="2018-01-01", end="2024-01-01")["Adj Close"]
returns = data.pct_change().dropna()

rolling_cov = returns.rolling(60).cov().dropna()

mu = returns.mean().values
cov = returns.cov().values
n = len(tickers)

w = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)

ret = mu @ w
risk = cp.quad_form(w, cov)
prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])

gammas = np.logspace(-2, 3, 50)
weights = []
rets = []
risks = []

for g in gammas:
    gamma.value = g
    prob.solve()
    w_opt = w.value
    weights.append(w_opt)
    risks.append(np.sqrt(w_opt.T @ cov @ w_opt))
    rets.append(mu @ w_opt)

plt.figure(figsize=(8, 5))
plt.plot(risks, rets)
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.grid(True)
plt.show()

mean_vec = returns.mean().values
cov_mat = returns.cov().values
simulations = 10000
horizon = 20
sim_results = []

for _ in range(simulations):
    rand = np.random.multivariate_normal(mean_vec, cov_mat, horizon)
    path = rand.mean(axis=1)
    sim_results.append(path.sum())

sim_results = np.array(sim_results)
var_95 = np.percentile(sim_results, 5)

print("Mean Returns:", mu)
print("Optimal Weights (last frontier point):", weights[-1])
print("Portfolio 95% VaR (Monte Carlo):", var_95)
