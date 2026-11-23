import yfinance as yf
import matplotlib.pyplot as plt

adj = yf.download(["AAPL", "MSFT"], start="2024-01-01", end="2024-06-30")["Adj Close"].dropna()

adj.plot(title="Adjusted Close: AAPL vs MSFT")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()

rets = adj.pct_change().dropna()
rets["AAPL"].plot(kind="hist", bins=40, alpha=0.6, edgecolor="black", label="AAPL")
rets["MSFT"].plot(kind="hist", bins=40, alpha=0.6, edgecolor="black", label="MSFT")
plt.title("Daily Returns Comparison")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print("Volatility AAPL:", rets["AAPL"].std())
print("Volatility MSFT:", rets["MSFT"].std())
