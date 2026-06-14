import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

tickers = ["AAPL", "MSFT", "AMZN", "META", "GOOG"]
data = yf.download(tickers, start="2024-01-01", end="2024-12-31", progress=False)

if data.empty:
    print("Download returned EMPTY data. Check internet, ticker symbols, or dates.")
    sys.exit(1)

price_cols = data.columns.get_level_values(0).unique().tolist()
use_col = "Adj Close" if "Adj Close" in price_cols else "Close"
prices = data[use_col].dropna(how="all")

prices = prices.loc[:, [c for c in tickers if c in prices.columns]]

if prices.shape[1] < 2:
    print("Not enough valid price columns after download. Try different dates/tickers.")
    sys.exit(1)

print("Using price column:", use_col)
print("Tickers loaded:", list(prices.columns))

returns = prices.pct_change().dropna()
print("\nReturns head():")
print(returns.head())

corr = returns.corr()
print("\nCorrelation matrix:")
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Daily Return Correlations (2024)")
plt.tight_layout()
plt.show()

pca = PCA(n_components=returns.shape[1])
pca.fit(returns.values)

pcs = pca.transform(returns.values)
pcs_df = pd.DataFrame(pcs, index=returns.index,
                      columns=[f"PC{i+1}" for i in range(returns.shape[1])])

loadings = pd.DataFrame(pca.components_.T,
                        index=returns.columns,
                        columns=[f"PC{i+1}" for i in range(returns.shape[1])])

print("\nExplained variance ratio (per PC):")
print(np.round(pca.explained_variance_ratio_, 4))

print("\nPC Loadings (eigenvectors):")
print(loadings)

plt.figure(figsize=(12, 5))
plt.plot(pcs_df.index, pcs_df["PC1"], label="PC1", linewidth=1)
plt.title("First Principal Component (PC1) of Returns")
plt.xlabel("Date"); plt.ylabel("PC1 Value")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
loadings["PC1"].plot(kind="bar")
plt.title("PC1 Loadings by Ticker")
plt.ylabel("Loading")
plt.grid(True, axis="y"); plt.tight_layout()
plt.show()
