import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["AAPL", "JPM", "XOM", "PG", "NVDA", "SPY"]

data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]
rets = np.log(data / data.shift(1)).dropna()

market = rets["SPY"]
stocks = rets.drop(columns="SPY")

capm_results = []
for s in stocks.columns:
    X = sm.add_constant(market)
    y = stocks[s]
    model = sm.OLS(y, X).fit()
    capm_results.append([s, model.params["const"], model.params["SPY"],
                         model.rsquared, model.rsquared_adj,
                         model.pvalues["SPY"]])

capm_df = pd.DataFrame(capm_results,
                       columns=["Ticker", "Alpha", "Beta (SPY)",
                                "R²", "Adj R²", "p(Beta)"])
print("\nCAPM results:")
print(capm_df.round(4))

multi_results = []
for s in stocks.columns:
    X = rets.drop(columns=s)          
    y = rets[s]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    multi_results.append([s, model.rsquared, model.rsquared_adj])

multi_df = pd.DataFrame(multi_results, columns=["Ticker", "R²", "Adj R²"])
print("\nMulti-factor results:")
print(multi_df.round(4))

X_factors = rets[["SPY", "AAPL", "JPM", "XOM", "PG", "NVDA"]].dropna()
X_vif = X_factors.drop(columns="SPY")
vif_df = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
})
print("\nVariance Inflation Factors:")
print(vif_df.round(2))

plt.figure(figsize=(8,5))
sns.barplot(y="Ticker", x="Beta (SPY)", data=capm_df, color="steelblue")
plt.title("Single-Factor CAPM Betas")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(y="Ticker", x="p(Beta)", data=capm_df, color="orange")
plt.title("CAPM p-values for Beta")
plt.axvline(0.05, color="red", ls="--", label="α = 0.05")
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="R²", y="Adj R²", data=multi_df, s=100)
plt.plot([0,1],[0,1],'k--')
plt.title("Multi-factor R² vs Adj R²")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x="Variable", y="VIF", data=vif_df, color="seagreen")
plt.axhline(5, color="red", ls="--")
plt.title("Variance Inflation Factors (VIF)")
plt.show()
