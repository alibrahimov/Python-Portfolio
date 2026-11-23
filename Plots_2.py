import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "AMZN", "META", "GOOG"]
data = yf.download(tickers, start = "2024-01-01", end="2024-12-31")
returns = data.pct_change().dropna()

plt.figure(figsize=(12,6))
for col in data.columns:
    plt.plot(data.index, data[col], label=col)
plt.title("Stock Prices in 2024")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()
