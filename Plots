import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys

df = yf.download("AAPL", start="2021-01-01", end="2023-12-31", progress=False)

if df.empty:
    print("Download returned an EMPTY DataFrame. Check your internet, dates, or ticker.")
    sys.exit(1)

print("Columns:", list(df.columns))

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
s = df[price_col].dropna()

df["Daily Return"] = s.pct_change()
df["MA5"]  = s.rolling(5).mean()
df["MA20"] = s.rolling(20).mean()

ax = s.plot(label=price_col, figsize=(12, 6), linewidth=1)
df["MA5"].plot(ax=ax, label="MA5")
df["MA20"].plot(ax=ax, label="MA20")

ax.set_title(f"AAPL {price_col} with 5/20-Day Moving Averages (2021–2023)")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
