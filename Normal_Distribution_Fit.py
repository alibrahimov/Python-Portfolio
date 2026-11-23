import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

data = yf.download("SPY", start="2020-01-01", end="2025-01-01")
price = data["Close"]

returns = np.log(price / price.shift(1)).dropna()
returns.head()

mu, sigma = norm.fit(returns) 
print(f"Fitted μ (mean)  = {mu:.6f}")
print(f"Fitted σ (std)   = {sigma:.6f}")

plt.figure(figsize=(8,5))
count, bins, ignored = plt.hist(returns, bins=60, density=True,
                                color="skyblue", alpha=0.6, edgecolor="k", label="Returns histogram")

x = np.linspace(returns.min(), returns.max(), 1000)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf, "r", lw=2, label=f"Normal PDF\nμ={mu:.5f}, σ={sigma:.5f}")

plt.title("SPY Daily Returns & Fitted Normal Distribution")
plt.xlabel("Daily return")
plt.ylabel("Density")
plt.legend()
plt.show()
